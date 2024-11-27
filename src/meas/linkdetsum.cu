
#include <cuda_common.h>
#include <meas/linkdetsum.h>
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

#include <reduce_block_1d.h>

using namespace std;


namespace CULQCD{
	
/////////////////////////////////////////////////////////////////////////////////////////
//////// Gauge determinant
/////////////////////////////////////////////////////////////////////////////////////////
//#ifdef USE_CUDA_CUB
template <int blockSize, bool UseTex, ArrayType atype, class Real> 
__global__ void kernel_linkdetsum(DetArg<Real> arg){
	int idd = INDEX1D();
	typedef cub::BlockReduce<complex, blockSize> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	complex res = complex::zero();

	if(idd < param_Volume()) {
		#ifdef MULTI_GPU
		int oddbit = 0;
		int id = idd;
		if(idd >= DEVPARAMS::HalfVolume){
			oddbit = 1;
			id = idd - DEVPARAMS::HalfVolume;
		}
		int x[NDIMS];
		Index_ND_EO(x, id, oddbit);
		for(int i=0; i<NDIMS;i++) x[i] += param_border(i);
		id = Index_ND_NM(x, DEVPARAMS::GridWGhost) >> 1;
		id += oddbit  * param_HalfVolumeG();
		int mustride = DEVPARAMS::VolumeG;
		int offset = mustride * NDIMS;
		#else
		int mustride = DEVPARAMS::Volume;
		int offset = DEVPARAMS::size;
		int id = idd;
		#endif
		//#pragma unroll
        for(int mu = 0; mu < NDIMS; mu++) res +=  GAUGE_LOAD<UseTex, atype, Real>( arg.array, id + mu * mustride, offset).det();
    }
	complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
	if (threadIdx.x == 0) CudaAtomicAdd(arg.value, aggregate);		
}


template <class Real> 
GaugeDetCUB<Real>::GaugeDetCUB(gauge &array):array(array){
	if(!array.EvenOdd()) errorCULQCD("gauge array must be a even/odd array...");
	value = complex::zero();
	arg.value = (complex*)dev_malloc(sizeof(complex));
	arg.array = array.GetPtr();
	size = 1;
	for(int i=0;i<NDIMS;i++) size *= PARAMS::Grid[i];
	timesec = 0.0;
}
template <class Real> 
GaugeDetCUB<Real>::~GaugeDetCUB(){ dev_free(arg.value); }
template <class Real> 
void GaugeDetCUB<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    CUDA_SAFE_CALL(cudaMemset(arg.value, 0, sizeof(complex)));
    if(PARAMS::UseTex){
    	//just ensure that the texture was not unbind somewhere...
    	BIND_GAUGE_TEXTURE(array.GetPtr());
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, true, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, true, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, true, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, true, SOA, Real);	
		#endif
	}
	else{
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, false, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, false, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, false, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_linkdetsum, tp, stream, arg, false, SOA, Real);	
		#endif
	}
}
template <class Real> 
complex GaugeDetCUB<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();//
	CUT_CHECK_ERROR("Link Determinant Sum: Kernel execution failed"); 
    CUDA_SAFE_CALL(cudaMemcpy(&value, arg.value, sizeof(complex), cudaMemcpyDeviceToHost));
	value /= (Real)(NDIMS * size);
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
complex GaugeDetCUB<Real>::Run(){
	return Run(0);
}
template <class Real> 
double GaugeDetCUB<Real>::time(){
	return timesec;
}
template <class Real> 
void GaugeDetCUB<Real>::stat(){
	COUT << "Link Determinant Sum:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}
template <class Real> 
void GaugeDetCUB<Real>::printValue(){
	printfCULQCD("Link Determinant Sum: < %.12e : %.12e >\n", value.real(), value.imag());
}



template <class Real> 
long long GaugeDetCUB<Real>::flop() const { 
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
long long GaugeDetCUB<Real>::bytes() const {
    //NEED TO RECOUNT!!!!!!  
	#ifdef MULTI_GPU
	return (2LL + array.getNumParams()) * 4LL * size * numnodes() * sizeof(Real);
	#else
	return (2LL + array.getNumParams()) * 4LL * size * sizeof(Real);
	#endif
}

template <class Real> 
double GaugeDetCUB<Real>::flops(){
	return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> 
double GaugeDetCUB<Real>::bandwidth(){
	return (double)bytes() / (timesec * (double)(1 << 30));
}


template class GaugeDetCUB<float>;
template class GaugeDetCUB<double>;
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//#else
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, ArrayType atype, class Real> 
__global__ void kernel_linkdetsum(DetArg<Real> arg){
	uint idd = INDEX1D();
    complex res = complex:: zero();
	if(idd < DEVPARAMS::Volume){
		#ifdef MULTI_GPU
		int oddbit = 0;
		int id = idd;
		if(idd >= DEVPARAMS::HalfVolume){
			oddbit = 1;
			id = idd - DEVPARAMS::HalfVolume;
		}
		int x[NDIMS];
		Index_ND_EO(x, id, oddbit);
		for(int i=0; i<NDIMS;i++) x[i] += param_border(i);
		id = Index_ND_NM(x, DEVPARAMS::GridWGhost) >> 1;
		id += oddbit  * param_HalfVolumeG();
		int mustride = DEVPARAMS::VolumeG;
		int offset = mustride * NDIMS;
		#else
		int mustride = DEVPARAMS::Volume;
		int offset = DEVPARAMS::size;
		int id = idd;
		#endif
		//#pragma unroll
        for(int mu = 0; mu < NDIMS; mu++) res +=  GAUGE_LOAD<UseTex, atype, Real>( arg.array, id + mu * mustride, offset).det();
    }		
	reduce_block_1d<complex>(arg.value, res);
}

template <class Real> 
GaugeDet<Real>::GaugeDet(gauge &array):array(array){
	value = complex::zero();
	arg.value = (complex*) dev_malloc(sizeof(complex));
	arg.array = array.GetPtr();
	size = 1;
	for(int i=0;i<NDIMS;i++) size *= PARAMS::Grid[i];
}
template <class Real> 
GaugeDet<Real>::~GaugeDet(){  dev_free(arg.value); }
template <class Real> 
void GaugeDet<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	size_t memshared = tp.block.x * sizeof(complex);
	CUDA_SAFE_CALL(cudaMemset(arg.value, 0, sizeof(complex)));
	if(array.EvenOdd()){
	    if(PARAMS::UseTex){
	    	//just ensure that the texture was not unbind somewhere...
	    	BIND_GAUGE_TEXTURE(array.GetPtr());
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_linkdetsum<true, SOA, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);		
	        if(array.Type() == SOA12) kernel_linkdetsum<true, SOA12, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);
	        if(array.Type() == SOA8) kernel_linkdetsum<true, SOA8, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);
			#else
	        kernel_linkdetsum<true, SOA, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);	
			#endif
	    }
	    else{
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_linkdetsum<false, SOA, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);
	        if(array.Type() == SOA12) kernel_linkdetsum<false, SOA12, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);
	        if(array.Type() == SOA8) kernel_linkdetsum<false, SOA8, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);
			#else
	        kernel_linkdetsum<false, SOA, Real><<<tp.grid,tp.block, memshared, stream>>>(arg);	
			#endif
	    }
	}
}
template <class Real> 
complex GaugeDet<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    apply(stream);
	CUDA_SAFE_CALL(cudaMemcpy(&value, arg.value, sizeof(complex), cudaMemcpyDeviceToHost));
	value /= (Real)(NDIMS * size);
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
complex GaugeDet<Real>::Run(){
	return Run(0);
}
template <class Real> 
double GaugeDet<Real>::time(){
	return timesec;
}
template <class Real> 
void GaugeDet<Real>::stat(){
	COUT << "Link Determinant Sum:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}
template <class Real> 
void GaugeDet<Real>::printValue(){
	printfCULQCD("Link Determinant Sum: < %.12e : %.12e >\n", value.real(), value.imag());
}



template <class Real> 
long long GaugeDet<Real>::flop() const { 
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
long long GaugeDet<Real>::bytes() const {
    //NEED TO RECOUNT!!!!!!  
	#ifdef MULTI_GPU
	return (2LL + array.getNumParams()) * 4LL * size * numnodes() * sizeof(Real);
	#else
	return (2LL + array.getNumParams()) * 4LL * size * sizeof(Real);
	#endif
}

template <class Real> 
double GaugeDet<Real>::flops(){
	return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> 
double GaugeDet<Real>::bandwidth(){
	return (double)bytes() / (timesec * (double)(1 << 30));
}


template class GaugeDet<float>;
template class GaugeDet<double>;
//#endif

}
