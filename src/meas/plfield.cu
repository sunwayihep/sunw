
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <ctime> // Needed for the true randomization
#include <cstdlib> 



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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////              Onepolyakovloop: only stores the complex trace          /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Real>
struct TPloopArg{
  complex *array;
  complex *ploop;
  complex *mean;
};

template <int blockSize, bool UseTex, ArrayType atypein, class Real>
__global__ void kernel_calc_Tpolyakovloop(TPloopArg<Real> arg){
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;  

	int id = INDEX1D();
  complex value = complex::zero();
	if(id < DEVPARAMS::tstride) {			
  	int index = id + (NDIMS-1) * DEVPARAMS::Volume;
  	int offset = DEVPARAMS::Volume * NDIMS;

  	msun L = GAUGE_LOAD<UseTex, atypein,Real>(arg.array, index, offset);
  	for( int t = 1; t < DEVPARAMS::Grid[NDIMS-1]; t++)
  		L *= GAUGE_LOAD<UseTex, atypein,Real>(arg.array, index + t * DEVPARAMS::tstride, offset);
    value = L.trace();
  	arg.ploop[id] = value;
  }
  complex aggregate = BlockReduce(temp_storage).Reduce(value, Summ<complex>());
  if(threadIdx.x==0) CudaAtomicAdd(arg.mean, aggregate);
}






template<bool UseTex, ArrayType atypein, class Real> 
class TPloop: Tunable{
private:
   gauge arrayin;
   TPloopArg<Real> arg;
   int size;
   complex value;
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
      CUDA_SAFE_CALL(cudaMemset(arg.mean, 0, sizeof(complex)));
      LAUNCH_KERNEL(kernel_calc_Tpolyakovloop, tp, stream, arg, UseTex, atypein, Real);
	}
public:
   TPloop(gauge &arrayin, complex *ploop):arrayin(arrayin){
		size = 1;
		//Number of threads is equal to the number of space points!
		for(int i=0;i<NDIMS-1;i++){
		  size *= PARAMS::Grid[i];
		} 
		timesec = 0.0;
    arg.array = arrayin.GetPtr();
    arg.ploop = ploop;
    arg.mean = (complex *)dev_malloc(sizeof(complex));
	}
  ~TPloop(){dev_free(arg.mean);};

	double time(){return timesec;}
	void stat(){ COUT << "MPloop:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
	long long flop() const { return 0;}
	long long bytes() const { return 0;}
	double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
	double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    for(int i=0; i<NDIMS-1; i++) vol << PARAMS::Grid[i] << "x";
    vol << PARAMS::Grid[NDIMS-1];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string typear = arrayin.ToStringArrayType();
    return TuneKey(vol.str().c_str(), typeid(*this).name(), typear.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
	void preTune() { }
	void postTune() {  }
	
	complex Run(const cudaStream_t &stream){
	#ifdef TIMMINGS
	    mtime.start();
	#endif
	  apply(stream);
    CUDA_SAFE_CALL(cudaMemcpy(&value, arg.mean, sizeof(complex), cudaMemcpyDeviceToHost));
    value /= (Real)(NCOLORS * size);
	#ifdef TIMMINGS
		CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Kernel execution failed");
	    mtime.stop();
	    timesec = mtime.getElapsedTimeInSec();
	#endif
      return value;
	}
	complex Run(){return Run(0);}
};



template<bool UseTex, ArrayType atypein, class Real>
complex TracePloop(gauge array, complex *ploop){
    TPloop<UseTex, atypein, Real> onepl(array, ploop);
    return onepl.Run();
}
/**
  @brief calculates the trace polyakov loop for each space point

*/
template<class Real>
complex TracePloop(gauge array, complex *ploop){
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
  const ArrayType atypein = SOA;
  complex value = complex::zero();
  if(PARAMS::UseTex){
   GAUGE_TEXTURE(array.GetPtr(), true);
   value = TracePloop<true, atypein, Real>(array, ploop);
  }
  else{
   value = TracePloop<false, atypein, Real>(array, ploop);
  }
  return value;
}
template complexs TracePloop<float>(gauges array, complexs *ploop);
template complexd TracePloop<double>(gauged array, complexd *ploop);






} //namespace CULQCD

