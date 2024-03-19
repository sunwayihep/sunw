
#include <cuda_common.h>
#include <meas/linkUF.h>
#include <matrixsun.h>
#include <complex.h>
#include <constants.h>
#include <index.h>
#include <device_load_save.h>
#include <reduction.h>
#include <timer.h>
#include <texture_host.h>
#include <modes.h>



#include <cudaAtomic.h>

#include <launch_kernel.cuh>

#include <cub/cub.cuh>

using namespace std;


namespace CULQCD{


	
/////////////////////////////////////////////////////////////////////////////////////////
//////// Gauge determinant
/////////////////////////////////////////////////////////////////////////////////////////
//#ifdef USE_CUDA_CUB
template <int blockSize, bool UseTex, ArrayType atype, class Real> 
__global__ void kernel_linkUF(GaugeUFArg<Real> arg){
	int id = INDEX1D();
	typedef cub::BlockReduce<complex, blockSize> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	complex res = complex::zero();

	if(id < param_Volume()) {
        
  
	  int x[4];
	  Index_4D_NM(id, x);
	  int mustride = DEVPARAMS::Volume;
	  int offset = DEVPARAMS::size;
  
  
	for(int mu = 0; mu < 4; mu++){
    int muvolume = mu * mustride;
	  msun link;
	  msun staple = msu3::zero();
	  for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
      int dx[4] = {0, 0, 0, 0};	
		  int nuvolume = nu * mustride;
		  link = GAUGE_LOAD<UseTex, atype, Real>( arg.array,  id + nuvolume, offset);
		  dx[nu]++;
		  link *= GAUGE_LOAD<UseTex, atype, Real>( arg.array, Index_4D_Neig_NM(x,dx) + muvolume, offset);	
		  dx[nu]--;
		  dx[mu]++;
		  link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.array, Index_4D_Neig_NM(x,dx) + nuvolume, offset);
		  staple += link;

		  dx[mu]--;
		  dx[nu]--;
		  link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.array,  Index_4D_Neig_NM(x,dx) + nuvolume, offset);	
		  link *= GAUGE_LOAD<UseTex, atype, Real>( arg.array, Index_4D_Neig_NM(x,dx)  + muvolume, offset);
		  dx[mu]++;
		  link *= GAUGE_LOAD<UseTex, atype, Real>( arg.array, Index_4D_Neig_NM(x,dx) + nuvolume, offset);
		  staple += link;
	  }
	  msun U = GAUGE_LOAD<UseTex, atype, Real>( arg.array,  id + muvolume, offset);
		res += (U * link.dagger()).trace();

	}
	}
	complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
	if (threadIdx.x == 0) CudaAtomicAdd(arg.value, aggregate);		
}


template <class Real> 
GaugeUFCUB<Real>::GaugeUFCUB(gauge &array):array(array){
	if(array.EvenOdd()) errorCULQCD("gauge array cannot be a even/odd array...");
	value = complex::zero();
	arg.value = (complex*)dev_malloc(sizeof(complex));
	arg.array = array.GetPtr();
	size = 1;
	for(int i=0;i<4;i++) size *= PARAMS::Grid[i];
	timesec = 0.0;
}
template <class Real> 
GaugeUFCUB<Real>::~GaugeUFCUB(){ dev_free(arg.value); }
template <class Real> 
void GaugeUFCUB<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    CUDA_SAFE_CALL(cudaMemset(arg.value, 0, sizeof(complex)));
    if(PARAMS::UseTex){
    	//just ensure that the texture was not unbind somewhere...
    	BIND_GAUGE_TEXTURE(array.GetPtr());
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, true, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, true, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, true, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, true, SOA, Real);	
		#endif
	}
	else{
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, false, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, false, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, false, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_linkUF, tp, stream, arg, false, SOA, Real);	
		#endif
	}
}
template <class Real> 
complex GaugeUFCUB<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();//
	CUT_CHECK_ERROR("Link Sum: Kernel execution failed"); 
    CUDA_SAFE_CALL(cudaMemcpy(&value, arg.value, sizeof(complex), cudaMemcpyDeviceToHost));
	value /= (Real)(4 * 3 * size);
	#ifdef MULTI_GPU
	comm_Allreduce(&value);
	value /= numnodes();
	#endif
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
	return value;
}

template <class Real> 
complex GaugeUFCUB<Real>::Run(){
	return Run(0);
}
template <class Real> 
double GaugeUFCUB<Real>::time(){
	return timesec;
}
template <class Real> 
void GaugeUFCUB<Real>::stat(){
	COUT << "Link Sum:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}
template <class Real> 
void GaugeUFCUB<Real>::printValue(){
	printfCULQCD("Link Sum: < %.12e : %.12e >\n", value.real(), value.imag());
}



template <class Real> 
long long GaugeUFCUB<Real>::flop() const { 
    //NEED TO RECOUNT!!!!!! 
#if (NCOLORS == 3)
	return (array.getNumFlop(true) + 82LL) * 4LL * size * numnodes();
#else
	unsigned int tmp_gs = 0;
	unsigned int tmp_det = 0;
	for(int i = 0; i<NCOLORS;i++){
        tmp_gs+=i+1;
        tmp_det+=i;
	}
	return (tmp_gs * NCOLORS * 8LL + tmp_det * (NCOLORS * 8LL + 11LL) ) * 4LL  * size * numnodes();
#endif
}
template <class Real> 
long long GaugeUFCUB<Real>::bytes() const {
    //NEED TO RECOUNT!!!!!!  
	#ifdef MULTI_GPU
	return (2LL + array.getNumParams()) * 4LL * size * numnodes() * sizeof(Real);
	#else
	return (2LL + array.getNumParams()) * 4LL * size * sizeof(Real);
	#endif
}

template <class Real> 
double GaugeUFCUB<Real>::flops(){
	return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> 
double GaugeUFCUB<Real>::bandwidth(){
	return (double)bytes() / (timesec * (double)(1 << 30));
}


template class GaugeUFCUB<float>;
template class GaugeUFCUB<double>;


}
