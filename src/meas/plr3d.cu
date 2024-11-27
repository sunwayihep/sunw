#if (NDIMS == 4) // compile only for 4D lattice
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <ctime> // Needed for the true randomization
#include <cstdlib> 


#include <meas/polyakovloop.h>
#include <cuda_common.h>
#include <constants.h>
#include <gaugearray.h>
#include <index.h>
#include <reduction.h>
#include <device_load_save.h>
#include <texture_host.h>
#include <timer.h>
#include <comm_mpi.h>


#include <tune.h>
#include <launch_kernel.cuh>


#include <cudaAtomic.h>

#include <cub/cub.cuh>

using namespace std;


namespace CULQCD{










template<class Real>
struct PotPLoop3DArg{
  complex *array;
  complex *pot;
  int radius;
  int pts;
};

template <class Real> 
class PotPLoop3D: Tunable{
private:
   PotPLoop3DArg<Real> arg;
   gauge array;
   int size;
   complex *pot;
   double timesec;
#ifdef TIMMINGS
    Timer PotPLoop3Dtime;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream);

public:
   PotPLoop3D(gauge &array, complex *pot, int radius, int pts);
   ~PotPLoop3D(){dev_free(arg.pot);};
   void Run(const cudaStream_t &stream);
   void Run();
   double flops();
   double bandwidth();
   long long flop() const ;
   long long bytes() const;
   double time();
   void stat();


  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    for(int i=0; i<NDIMS-1; i++) vol << PARAMS::Grid[i] << "x";
    vol << PARAMS::Grid[NDIMS-1];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), array.ToStringArrayType().c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() {  }

};




template <int blockSize, bool UseTex, ArrayType atypein, class Real>
__global__ void kernel_calc_polyakovloop3D(PotPLoop3DArg<Real> arg){
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

	int id = INDEX1D();
  msun plup = msun::zero();
  if(id < DEVPARAMS::tstride) plup = GAUGE_LOAD<UseTex, atypein,Real>(arg.array, id, DEVPARAMS::tstride);
  int x[NDIMS-1];
  Index_NDs_NM(id, x);
  for(int k = 0; k <= arg.radius; k++)
  for(int j = 0; j <= arg.radius; j++)
  for(int i = 0; i <= arg.radius; i++){
  	int idpts = i + j * (arg.radius+1) + k * (arg.radius+1)*(arg.radius+1);
    __syncthreads();
    complex pp = complex::zero();
    complex ppdagger = complex::zero();
    if(id < DEVPARAMS::tstride){	
		int dx[NDIMS-1] = {i,j,k};
	    msun pldown = GAUGE_LOAD<UseTex, atypein,Real>(arg.array, Index_NDs_Neig_NM(x, dx), DEVPARAMS::tstride);
	    pp = plup.trace() * pldown.trace();
	    ppdagger = plup.trace() * pldown.dagger().trace();
    }
    complex aggregate = BlockReduce(temp_storage).Reduce(ppdagger, Summ<complex>());
    if (threadIdx.x == 0) CudaAtomicAdd(arg.pot + idpts, aggregate);
    __syncthreads();
    aggregate = BlockReduce(temp_storage).Reduce(pp, Summ<complex>());
    if (threadIdx.x == 0) CudaAtomicAdd(arg.pot + idpts + arg.pts, aggregate);
    __syncthreads();
  }
}


template <class Real> 
PotPLoop3D<Real>::PotPLoop3D(gauge &array, complex *pot, int radius, int pts):array(array), pot(pot){
  if(array.EvenOdd() == true) errorCULQCD("Not defined for EvenOdd arrays...\n");

	size = 1;
	for(int i=0;i<NDIMS-1;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;
	arg.array = array.GetPtr();
	arg.pot = (complex *)dev_malloc( 2 * pts * sizeof(complex));
	arg.radius = radius;
	arg.pts = pts;
  
}
template <class Real> 
void PotPLoop3D<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  CUDA_SAFE_CALL(cudaMemset(arg.pot, 0, 2 * arg.pts * sizeof(complex)));
  if(PARAMS::UseTex){
  	//just ensure that the texture was not unbind somewhere...
  	BIND_GAUGE_TEXTURE(array.GetPtr());
	#if (NCOLORS == 3)
    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, true, SOA, Real);		
    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, true, SOA12, Real);
    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, true, SOA8, Real);
	#else
    LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, true, SOA, Real);	
	#endif
  }
  else{
	  #if (NCOLORS == 3)
      if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, false, SOA, Real);		
      if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, false, SOA12, Real);
      if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, false, SOA8, Real);
	  #else
      LAUNCH_KERNEL(kernel_calc_polyakovloop3D, tp, stream, arg, false, SOA, Real);	
	  #endif
  }
  
}

template <class Real> 
void PotPLoop3D<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    PotPLoop3Dtime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(pot, arg.pot, 2 * arg.pts * sizeof(complex), cudaMemcpyDeviceToHost));
    //normalize!!!!!!
    for(int r = 0; r < 2 * arg.pts; r++)
      pot[r] /= (Real)(NCOLORS * NCOLORS * size);

#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    PotPLoop3Dtime.stop();
    timesec = PotPLoop3Dtime.getElapsedTimeInSec();
#endif
}
template <class Real> 
void PotPLoop3D<Real>::Run(){	return Run(0);}
template <class Real> 
double PotPLoop3D<Real>::time(){	return timesec;}
template <class Real> 
void PotPLoop3D<Real>::stat(){	COUT << "PotPLoop3D:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}


template <class Real> 
long long PotPLoop3D<Real>::flop() const {return 0;}
template <class Real> 
long long PotPLoop3D<Real>::bytes() const {	return 0;}
template <class Real> 
double PotPLoop3D<Real>::flops(){	return ((double)flop() * 1.0e-9) / timesec;}
template <class Real> 
double PotPLoop3D<Real>::bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}










/*pot array size:
  radius = 8;
  int pts = (radius+1) * (radius+1) * (radius+1);
  complex *pot = (complex*) safe_malloc( 2 * pts * sizeof(complex));
  */


template<class Real>
void PotPL3D(gauge ploop, complex *pot, int radius, int pts){
  if(ploop.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(ploop.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  PotPLoop3D<Real> pl(ploop, pot, radius, pts);
  pl.Run();
  pl.stat();
}
template void PotPL3D<float>(gauges array, complexs *pot, int radius, int pts);
template void PotPL3D<double>(gauged array, complexd *pot, int radius, int pts);













} //namespace CULQCD
#endif
