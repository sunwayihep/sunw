
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////                                                                   Onepolyakovloop                                                                                          /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real>
__global__ void kernel_calc_Mpolyakovloop(complex *array, complex *ploop){
	int id = INDEX1D();
	if(id < DEVPARAMS::tstride){			
		int index = id + 3 * DEVPARAMS::Volume;
		int offset = DEVPARAMS::Volume * 4;

		msun L = GAUGE_LOAD<UseTex, atypein,Real>(array, index, offset);
		for( int t = 1; t < DEVPARAMS::Grid[3]; t++)
			L *= GAUGE_LOAD<UseTex, atypein,Real>(array, index + t * DEVPARAMS::tstride, offset);
			
		GAUGE_SAVE<atypeout, Real>(ploop, L, id, DEVPARAMS::tstride );
	}
}






template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real> 
class MPloop: Tunable{
private:
   gauge arrayin;
   gauge arrayout;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer mtime;
#endif
   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_calc_Mpolyakovloop<UseTex, atypein, atypeout, Real><<<tp.grid,tp.block, 0, stream>>>(arrayin.GetPtr(), arrayout.GetPtr());
	}
public:
   MPloop(gauge &arrayin, gauge &arrayout):arrayin(arrayin), arrayout(arrayout){
		size = 1;
		//Number of threads is equal to the number of space points!
		for(int i=0;i<3;i++){
		  size *= PARAMS::Grid[i];
		} 
		timesec = 0.0;
	}
  ~MPloop(){};

	double time(){return timesec;}
	void stat(){ COUT << "MPloop:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
	long long flop() const { return 0;}
	long long bytes() const { return 0;}
	double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
	double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string typear = arrayin.ToStringArrayType() + arrayout.ToStringArrayType();
    return TuneKey(vol.str().c_str(), typeid(*this).name(), typear.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
	void preTune() { arrayin.Backup(); }
	void postTune() { arrayin.Restore(); }
	
	void Run(const cudaStream_t &stream){
	#ifdef TIMMINGS
	    mtime.start();
	#endif
	    apply(stream);
	#ifdef TIMMINGS
		CUDA_SAFE_DEVICE_SYNC( );
	    mtime.stop();
	    timesec = mtime.getElapsedTimeInSec();
	#endif
	}
	void Run(){return Run(0);}
};















template<class Real>
struct PotPLoopArg{
  complex *array;
  complex *pot;
  int radius;
};

template <class Real, bool ppdagger> 
class PotPLoop: Tunable{
private:
   PotPLoopArg<Real> arg;
   gauge array;
   int size;
   complex *pot;
   double timesec;
#ifdef TIMMINGS
    Timer PotPLooptime;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream);

public:
   PotPLoop(gauge &array, complex *pot, int radius);
   ~PotPLoop(){dev_free(arg.pot);};
   void Run(const cudaStream_t &stream);
   void Run();
   double flops();
   double bandwidth();
   long long flop() const ;
   long long bytes() const;
   double time();
   void stat();
   void printValue();


  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
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



		

template <int blockSize, bool UseTex, ArrayType atypein, class Real, bool ppdagger>
__global__ void kernel_calc_polyakovloop(PotPLoopArg<Real> arg){
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

	int id = INDEX1D();
  msun plup;
  if(id < DEVPARAMS::tstride)
    plup = GAUGE_LOAD<UseTex, atypein,Real>(arg.array, id, DEVPARAMS::tstride);
  for(int r = 0; r <= arg.radius; r++){
    complex value = complex::zero();
    if(id < DEVPARAMS::tstride){	
      for(int mu = 0; mu < 3; mu++){
        msun pldown = GAUGE_LOAD<UseTex, atypein,Real>(arg.array, Index_3D_Neig_NM(id, mu, r), DEVPARAMS::tstride);;
        if( ppdagger ) value += plup.trace() * pldown.dagger().trace();
        else value += plup.trace() * pldown.trace();
      }
    }
    complex aggregate = BlockReduce(temp_storage).Reduce(value, Summ<complex>());
    if (threadIdx.x == 0) CudaAtomicAdd(arg.pot + r, aggregate);
  }
}


template <class Real, bool ppdagger> 
PotPLoop<Real, ppdagger>::PotPLoop(gauge &array, complex *pot, int radius):array(array), pot(pot){
  if(array.EvenOdd() == true) errorCULQCD("Not defined for EvenOdd arrays...\n");

	size = 1;
	for(int i=0;i<3;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;
	arg.array = array.GetPtr();
	arg.pot = (complex *)dev_malloc( (radius+1) * sizeof(complex));
	arg.radius = radius;
  
}
template <class Real, bool ppdagger> 
void PotPLoop<Real, ppdagger>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  CUDA_SAFE_CALL(cudaMemset(arg.pot, 0, (arg.radius+1) * sizeof(complex)));
  if(PARAMS::UseTex){
  	//just ensure that the texture was not unbind somewhere...
  	BIND_GAUGE_TEXTURE(array.GetPtr());
	#if (NCOLORS == 3)
    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA, Real, ppdagger);		
    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA12, Real, ppdagger);
    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA8, Real, ppdagger);
	#else
    LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA, Real, ppdagger);	
	#endif
  }
  else{
	  #if (NCOLORS == 3)
      if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA, Real, ppdagger);		
      if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA12, Real, ppdagger);
      if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA8, Real, ppdagger);
	  #else
      LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA, Real, ppdagger);	
	  #endif
  }
  
}

template <class Real, bool ppdagger> 
void PotPLoop<Real, ppdagger>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    PotPLooptime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(pot, arg.pot, (arg.radius+1)*sizeof(complex), cudaMemcpyDeviceToHost));
    //normalize!!!!!!
    for(int r = 0; r <= arg.radius; r++)
      pot[r] /= (Real)(3 * NCOLORS * NCOLORS * size);

#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    PotPLooptime.stop();
    timesec = PotPLooptime.getElapsedTimeInSec();
#endif
}
template <class Real, bool ppdagger> 
void PotPLoop<Real, ppdagger>::Run(){	return Run(0);}
template <class Real, bool ppdagger> 
double PotPLoop<Real, ppdagger>::time(){	return timesec;}
template <class Real, bool ppdagger> 
void PotPLoop<Real, ppdagger>::stat(){	COUT << "PotPLoop:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
template <class Real, bool ppdagger> 
void PotPLoop<Real, ppdagger>::printValue(){	
  for(int r = 0; r <= arg.radius; r++)
	  printfCULQCD("R = %d: < %.12e : %.12e : %.12e >\n", r,  pot[r].real(), pot[r].imag(), pot[r].abs());
}
template <class Real, bool ppdagger> 
long long PotPLoop<Real, ppdagger>::flop() const {return 0;}
template <class Real, bool ppdagger> 
long long PotPLoop<Real, ppdagger>::bytes() const {	return 0;}
template <class Real, bool ppdagger> 
double PotPLoop<Real, ppdagger>::flops(){	return ((double)flop() * 1.0e-9) / timesec;}
template <class Real, bool ppdagger> 
double PotPLoop<Real, ppdagger>::bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}













template<class Real>
void PotPL(gauge ploop, complex *pot, int radius, bool ppdagger){
  if(ploop.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(ploop.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
  if(ppdagger){
    PotPLoop<Real, true> pl(ploop, pot, radius);
    pl.Run();
    pl.printValue();
  }
  else{
    PotPLoop<Real, false> pl(ploop, pot, radius);
    pl.Run();
    pl.printValue();
  }
}
template void PotPL<float>(gauges array, complexs *pot, int radius, bool ppdagger);
template void PotPL<double>(gauged array, complexd *pot, int radius, bool ppdagger);













} //namespace CULQCD

