
#include <cuda_common.h>
#include <meas/linktrsum.h>
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
//////// Gauge Trace
/////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, ArrayType atype, class Real> 
inline __device__ complex dev_linktracesum(complex *array, int id, int offset){
	complex sumtrace = complex::zero();  
#if (NCOLORS >3)
#pragma unroll
	for(int i = 0; i< NCOLORS; i++)
		sumtrace +=ELEM_LOAD<UseTex, Real>(array, id + (i + i * NCOLORS) * offset); 
#else
	if(atype == SOA){ 
#pragma unroll
		for(int i = 0; i< NCOLORS; i++)
			sumtrace +=ELEM_LOAD<UseTex, Real>(array, id + (i + i * NCOLORS) * offset); 
	} 
	if(atype == SOA12){
		complex tmp[4];  
		tmp[0] =ELEM_LOAD<UseTex, Real>(array, id);
		tmp[1] =ELEM_LOAD<UseTex, Real>(array, id + 4 * offset);
		tmp[2] =ELEM_LOAD<UseTex, Real>(array, id + offset);
		tmp[3] =ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);
		sumtrace +=tmp[0];
		sumtrace +=tmp[1];
		sumtrace += ~(tmp[0] * tmp[1] - tmp[2] * tmp[3]);
	}
	if(atype == SOA8){
		msun tmplink;
		tmplink.e[0][1] =ELEM_LOAD<UseTex, Real>(array, id);
		tmplink.e[0][2] =ELEM_LOAD<UseTex, Real>(array, id + offset);
		tmplink.e[1][0] =ELEM_LOAD<UseTex, Real>(array, id + 2 * offset);
		complex theta =ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);	
		reconstruct8p<Real>(tmplink, theta);	
		sumtrace = tmplink.trace();
	}
#endif
	return sumtrace;
}



//#ifdef USE_CUDA_CUB
template <int blockSize, bool UseTex, ArrayType atype, class Real> 
__global__ void kernel_linktracesum(TraceArg<Real> arg){
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
    complex res = complex::zero();
	uint idd = INDEX1D();
	if(idd < DEVPARAMS::Volume){
		#ifdef MULTI_GPU
		int oddbit = 0;
		int id = idd;
		if(idd >= DEVPARAMS::HalfVolume){
			oddbit = 1;
			id = idd - DEVPARAMS::HalfVolume;
		}
			int x[4];
			Index_4D_EO(x, id, oddbit);
			for(int i=0; i<4;i++) x[i] += param_border(i);
			id = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) >> 1 ;
			id += oddbit  * param_HalfVolumeG();
			int mustride = DEVPARAMS::VolumeG;
			int offset = mustride * 4;
		#else
			int mustride = DEVPARAMS::Volume;
			int offset = DEVPARAMS::size;
			int id = idd;
		#endif
		#pragma unroll
	    for(int mu = 0; mu < 4; mu++) res += dev_linktracesum<UseTex, atype, Real>( arg.array, id + mu * mustride, offset);
	}		
  complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
  if (threadIdx.x == 0) CudaAtomicAdd(arg.value, aggregate);
}



template <class Real> 
GaugeTraceCUB<Real>::GaugeTraceCUB(gauge &array):array(array){
	if(!array.EvenOdd()) errorCULQCD("gauge array must be a even/odd array...");
	value = complex::zero();
	arg.value = (complex*)dev_malloc(sizeof(complex));
	arg.array = array.GetPtr();
	size = 1;
	for(int i=0;i<4;i++) size *= PARAMS::Grid[i];
	timesec = 0.0;
}
template <class Real> 
GaugeTraceCUB<Real>::~GaugeTraceCUB(){ dev_free(arg.value); }
template <class Real> 
void GaugeTraceCUB<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    CUDA_SAFE_CALL(cudaMemset(arg.value, 0, sizeof(complex)));
    if(PARAMS::UseTex){
    	//just ensure that the texture was not unbind somewhere...
    	BIND_GAUGE_TEXTURE(array.GetPtr());
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, true, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, true, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, true, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, true, SOA, Real);	
		#endif
	}
	else{
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, false, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, false, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, false, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_linktracesum, tp, stream, arg, false, SOA, Real);	
		#endif
	}
}
template <class Real> 
complex GaugeTraceCUB<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();//
	CUT_CHECK_ERROR("Link Determinant Sum: Kernel execution failed"); 
    CUDA_SAFE_CALL(cudaMemcpy(&value, arg.value, sizeof(complex), cudaMemcpyDeviceToHost));
	value /= (Real)(4 * NCOLORS * size);
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
complex GaugeTraceCUB<Real>::Run(){
	return Run(0);
}
template <class Real> 
double GaugeTraceCUB<Real>::time(){
	return timesec;
}
template <class Real> 
void GaugeTraceCUB<Real>::stat(){
	COUT << "Link Trace Sum:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}
template <class Real> 
void GaugeTraceCUB<Real>::printValue(){
	printfCULQCD("Link Trace Sum: < %.12e : %.12e >\n", value.real(), value.imag());
}
template <class Real> 
long long GaugeTraceCUB<Real>::flop() const { 
#if (NCOLORS == 3)
	if(array.Type() == SOA){
		return (NCOLORS * 2LL + 8) * size * numnodes();
	}
	else if(array.Type() == SOA12){
		return (18LL + 8) * size * numnodes();
	}
	else if(array.Type() == SOA8){
		return (array.getNumFlop(true)  + 8LL) * size * numnodes();
	}
	else return 0;
#else
	return (NCOLORS * 2LL + 8) * size * numnodes();
#endif
}
template <class Real> 
long long GaugeTraceCUB<Real>::bytes() const { 
#if (NCOLORS == 3)
	if(array.Type() == SOA){
		return (NCOLORS * 8LL + 2LL) * size * numnodes() * sizeof(Real);
	}
	else if(array.Type() == SOA12){
		return (32 + 2) * size * numnodes() * sizeof(Real);
	}
	else if(array.Type() == SOA8){
		return(8 + 2LL) * size * numnodes() * sizeof(Real);
	}
	else return 0;
#else
	return (NCOLORS * 8LL + 2LL) * size * numnodes() * sizeof(Real);
#endif
}
template <class Real> 
double GaugeTraceCUB<Real>::flops(){
	return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> 
double GaugeTraceCUB<Real>::bandwidth(){
	return (double)bytes() / (timesec * (double)(1 << 30));
}

template <class Real> 
  TuneKey GaugeTraceCUB<Real>::tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), array.ToStringArrayType().c_str(), aux.str().c_str());
  }
template <class Real> 
  std::string GaugeTraceCUB<Real>::paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
return ps.str();
}



template class GaugeTraceCUB<float>;
template class GaugeTraceCUB<double>;
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//#else
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <bool UseTex, ArrayType atype, class Real> 
__global__ void kernel_linktracesum(TraceArg<Real> arg){
	uint idd = INDEX1D();
	if(idd < DEVPARAMS::Volume){
		#ifdef MULTI_GPU
		int oddbit = 0;
		int id = idd;
		if(idd >= DEVPARAMS::HalfVolume){
			oddbit = 1;
			id = idd - DEVPARAMS::HalfVolume;
		}
			int x[4];
			Index_4D_EO(x, id, oddbit);
			for(int i=0; i<4;i++) x[i] += param_border(i);
			id = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1] ) * param_GridG(0) + x[0]) >> 1 ;
			id += oddbit  * param_HalfVolumeG();
			int mustride = DEVPARAMS::VolumeG;
			int offset = mustride * 4;
		#else
			int mustride = DEVPARAMS::Volume;
			int offset = DEVPARAMS::size;
			int id = idd;
		#endif
        complex res = complex:: zero();
		#pragma unroll
        for(int mu = 0; mu < 4; mu++) res += dev_linktracesum<UseTex, atype, Real>( arg.array, id + mu * mustride, offset);
        arg.value[idd] = res;
    }		
}

template <class Real> 
GaugeTrace<Real>::GaugeTrace(gauge &array, complex *sum):array(array){
	value = complex::zero();
	arg.value = sum;
	arg.array = array.GetPtr();
	size = 1;
	for(int i=0;i<4;i++) size *= PARAMS::Grid[i];
	timesec = 0.0;
}
template <class Real> 
GaugeTrace<Real>::~GaugeTrace(){  }
template <class Real> 
void GaugeTrace<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if(array.EvenOdd()){
	    if(PARAMS::UseTex){
	    	//just ensure that the texture was not unbind somewhere...
	    	BIND_GAUGE_TEXTURE(array.GetPtr());
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_linktracesum<true, SOA, Real><<<tp.grid,tp.block, 0, stream>>>(arg);		
	        if(array.Type() == SOA12) kernel_linktracesum<true, SOA12, Real><<<tp.grid,tp.block, 0, stream>>>(arg);
	        if(array.Type() == SOA8) kernel_linktracesum<true, SOA8, Real><<<tp.grid,tp.block, 0, stream>>>(arg);
			#else
	        kernel_linktracesum<true, SOA, Real><<<tp.grid,tp.block, 0, stream>>>(arg);	
			#endif
	    }
	    else{
			#if (NCOLORS == 3)
	        if(array.Type() == SOA) kernel_linktracesum<false, SOA, Real><<<tp.grid,tp.block, 0, stream>>>(arg);
	        if(array.Type() == SOA12) kernel_linktracesum<false, SOA12, Real><<<tp.grid,tp.block, 0, stream>>>(arg);
	        if(array.Type() == SOA8) kernel_linktracesum<false, SOA8, Real><<<tp.grid,tp.block, 0, stream>>>(arg);
			#else
	        kernel_linktracesum<false, SOA, Real><<<tp.grid,tp.block, 0, stream>>>(arg);	
			#endif
	    }
	}
}
template <class Real> 
complex GaugeTrace<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    apply(stream);
	value = reduction<complex>(arg.value, size, stream);
	value /= (Real)(4 * NCOLORS * size);
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
complex GaugeTrace<Real>::Run(){
	return Run(0);
}
template <class Real> 
double GaugeTrace<Real>::time(){
	return timesec;
}
template <class Real> 
void GaugeTrace<Real>::stat(){
	COUT << "Link Trace Sum:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}
template <class Real> 
void GaugeTrace<Real>::printValue(){
	printfCULQCD("Link Trace Sum: < %.12e : %.12e >\n", value.real(), value.imag());
}
template <class Real> 
long long GaugeTrace<Real>::flop() const { 
#if (NCOLORS == 3)
	if(array.Type() == SOA){
		return (NCOLORS * 2LL + 8) * size * numnodes();
	}
	else if(array.Type() == SOA12){
		return (18LL + 8) * size * numnodes();
	}
	else if(array.Type() == SOA8){
		return (array.getNumFlop(true)  + 8LL) * size * numnodes();
	}
	else return 0;
#else
	return (NCOLORS * 2LL + 8) * size * numnodes();
#endif
}
template <class Real> 
long long GaugeTrace<Real>::bytes() const { 
#if (NCOLORS == 3)
	if(array.Type() == SOA){
		return (NCOLORS * 8LL + 2LL) * size * numnodes() * sizeof(Real);
	}
	else if(array.Type() == SOA12){
		return (32 + 2) * size * numnodes() * sizeof(Real);
	}
	else if(array.Type() == SOA8){
		return(8 + 2LL) * size * numnodes() * sizeof(Real);
	}
	else return 0;
#else
	return (NCOLORS * 8LL + 2LL) * size * numnodes() * sizeof(Real);
#endif
}
template <class Real> 
double GaugeTrace<Real>::flops(){
	return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> 
double GaugeTrace<Real>::bandwidth(){
	return (double)bytes() / (timesec * (double)(1 << 30));
}

template <class Real> 
  TuneKey GaugeTrace<Real>::tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), array.ToStringArrayType().c_str(), aux.str().c_str());
  }
template <class Real> 
  std::string GaugeTrace<Real>::paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
return ps.str();
}

template class GaugeTrace<float>;
template class GaugeTrace<double>;

//#endif


}
