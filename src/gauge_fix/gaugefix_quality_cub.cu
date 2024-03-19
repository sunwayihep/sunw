
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <typeinfo>
	

#include <cuda_common.h>
#include <comm_mpi.h>
#include <complex.h>
#include <matrixsun.h>

#include <tune.h>
#include <index.h>
#include <device_load_save.h>
#include <texture.h>
#include <texture_host.h>
#include <timer.h>
#include <constants.h>
#include <modes.h>



#include <gaugefix/gaugefix.h>
#include <cudaAtomic.h>
#include <cub/cub.cuh>
#include <launch_kernel.cuh>

using namespace std;



namespace CULQCD{



#ifdef USE_GAUGE_FIX


template<int blockSize, int DIR, bool UseTex, ArrayType atype, class Real>
__global__ void kernel_gaugefix_quality_cub(GaugeFixQualityArg<Real> arg){
	typedef cub::BlockReduce<complex, blockSize> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	complex res = complex::zero();

	int idd = INDEX1D();
	if(idd < DEVPARAMS::Volume){
		int id = idd;
		int oddbit = 0;
		if(idd >= DEVPARAMS::HalfVolume){
			id -= DEVPARAMS::HalfVolume;
			oddbit = 1;
		}
		int offset = DEVPARAMS::VolumeG * 4;
		int idx = EOIndeX(id, oddbit);
		msun delta = msun::zero();
		//Uplinks
		for(int nu = 0; nu < DIR; nu++) 
			delta -= GAUGE_LOAD<UseTex, atype, Real>( arg.array,  idx + nu * DEVPARAMS::VolumeG, offset);
		//Fg (sum_DIR uplinks)
		res.real() = -delta.realtrace();
		//Downlinks
		for(int nu = 0; nu < DIR; nu++) 	
			delta += GAUGE_LOAD<UseTex, atype, Real>( arg.array, neighborEOIndexMinusOne(id, oddbit, nu) + nu * DEVPARAMS::VolumeG, offset);
		delta = (delta-delta.dagger()).subtraceunit();
		//theta
		res.imag() = realtraceUVdagger(delta, delta);
	}
	complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
	if (threadIdx.x == 0) CudaAtomicAdd(arg.value, aggregate);		
}

template<int DIR, bool UseTex, ArrayType atype, class Real>
GaugeFixQualityCUB<DIR, UseTex, atype, Real>::GaugeFixQualityCUB(gauge &array):array(array){
	if(array.Type() != atype) errorCULQCD("gauge array type and template types do not match...");
	if(!array.EvenOdd()) errorCULQCD("gauge array must be a even/odd array...");
	functionName = "GaugeFixQualityCUB";
	value = complex::zero();
	arg.value = (complex*)dev_malloc(sizeof(complex));
	arg.array = array.GetPtr();
	size = 1;
	for(int i=0;i<4;i++){
		grid[i]=PARAMS::Grid[i];
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;
}

template<int DIR, bool UseTex, ArrayType atype, class Real>
void GaugeFixQualityCUB<DIR, UseTex, atype, Real>::apply(const cudaStream_t &stream){
  TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  CUDA_SAFE_CALL(cudaMemset(arg.value, 0, sizeof(complex)));
  LAUNCH_KERNEL(kernel_gaugefix_quality_cub, tp, stream, arg, DIR, UseTex, atype, Real);
}
template<int DIR, bool UseTex, ArrayType atype, class Real>
GaugeFixQualityCUB<DIR, UseTex, atype, Real>::~GaugeFixQualityCUB(){dev_free(arg.value);};
template<int DIR, bool UseTex, ArrayType atype, class Real>
complex GaugeFixQualityCUB<DIR, UseTex, atype, Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    mtime.start();
#endif
    if(UseTex){
    	BIND_GAUGE_TEXTURE(array.GetPtr());
    }
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();//
	CUT_CHECK_ERROR("Kernel execution failed"); 
    CUDA_SAFE_CALL(cudaMemcpy(&value, arg.value, sizeof(complex), cudaMemcpyDeviceToHost));
	value /= (Real)(PARAMS::Volume * NCOLORS);
	value.real() /= (Real)DIR;
	#ifdef MULTI_GPU
	comm_Allreduce(&value);
	value /= numnodes();
	#endif
#ifdef TIMMINGS
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
	return value;
}
template<int DIR, bool UseTex, ArrayType atype, class Real>
complex GaugeFixQualityCUB<DIR, UseTex, atype, Real>::Run(){return Run(0);}
template<int DIR, bool UseTex, ArrayType atype, class Real>
double GaugeFixQualityCUB<DIR, UseTex, atype, Real>::flops(){	return ((double)flop() * 1.0e-9) / timesec;}
template<int DIR, bool UseTex, ArrayType atype, class Real>
double GaugeFixQualityCUB<DIR, UseTex, atype, Real>::bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
template<int DIR, bool UseTex, ArrayType atype, class Real>
long long GaugeFixQualityCUB<DIR, UseTex, atype, Real>::flop() const { 
	long long arrayflops = 2LL * DIR * array.getNumFlop(true);
	return (arrayflops + 2LL * NCOLORS * NCOLORS * (DIR + 1) + 4LL * NCOLORS * ( 1 + NCOLORS) ) * size * numnodes();
}
template<int DIR, bool UseTex, ArrayType atype, class Real>
long long GaugeFixQualityCUB<DIR, UseTex, atype, Real>::bytes() const { 
	return (2LL * DIR * array.getNumParams() + 2LL) * size * sizeof(Real) * numnodes();
}

template<int DIR, bool UseTex, ArrayType atype, class Real>
double GaugeFixQualityCUB<DIR, UseTex, atype, Real>::time(){return timesec;}

template<int DIR, bool UseTex, ArrayType atype, class Real>
void GaugeFixQualityCUB<DIR, UseTex, atype, Real>::stat(){
COUT << "GaugeFixQualityCUB:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
}
template<int DIR, bool UseTex, ArrayType atype, class Real>
void GaugeFixQualityCUB<DIR, UseTex, atype, Real>::printValue(){
	printfCULQCD("GaugeFixQualityCUB:Fg = %.12e\ttheta = %.12e\n", value.real(), value.imag() );
}




#if (NCOLORS == 3)
template class GaugeFixQualityCUB<4, true, SOA, float>;
template class GaugeFixQualityCUB<4, true, SOA12, float>;
template class GaugeFixQualityCUB<4, true, SOA8, float>;

template class GaugeFixQualityCUB<4, true, SOA, double>;
template class GaugeFixQualityCUB<4, true, SOA12, double>;
template class GaugeFixQualityCUB<4, true, SOA8, double>;


template class GaugeFixQualityCUB<4, false, SOA, float>;
template class GaugeFixQualityCUB<4, false, SOA12, float>;
template class GaugeFixQualityCUB<4, false, SOA8, float>;

template class GaugeFixQualityCUB<4, false, SOA, double>;
template class GaugeFixQualityCUB<4, false, SOA12, double>;
template class GaugeFixQualityCUB<4, false, SOA8, double>;


template class GaugeFixQualityCUB<3, true, SOA, float>;
template class GaugeFixQualityCUB<3, true, SOA12, float>;
template class GaugeFixQualityCUB<3, true, SOA8, float>;

template class GaugeFixQualityCUB<3, true, SOA, double>;
template class GaugeFixQualityCUB<3, true, SOA12, double>;
template class GaugeFixQualityCUB<3, true, SOA8, double>;


template class GaugeFixQualityCUB<3, false, SOA, float>;
template class GaugeFixQualityCUB<3, false, SOA12, float>;
template class GaugeFixQualityCUB<3, false, SOA8, float>;

template class GaugeFixQualityCUB<3, false, SOA, double>;
template class GaugeFixQualityCUB<3, false, SOA12, double>;
template class GaugeFixQualityCUB<3, false, SOA8, double>;
#elif (NCOLORS > 3)
template class GaugeFixQualityCUB<4, true, SOA, float>;
template class GaugeFixQualityCUB<4, true, SOA, double>;

template class GaugeFixQualityCUB<4, false, SOA, float>;
template class GaugeFixQualityCUB<4, false, SOA, double>;


template class GaugeFixQualityCUB<3, true, SOA, float>;
template class GaugeFixQualityCUB<3, true, SOA, double>;

template class GaugeFixQualityCUB<3, false, SOA, float>;
template class GaugeFixQualityCUB<3, false, SOA, double>;
#else
#error Code not done for NCOLORS < 3
#endif

#endif

}
