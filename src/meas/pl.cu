
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
	void preTune() { }
	void postTune() {  }
	
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



/**
  @brief calculates the polyakov loop for each space point

*/
template<class Real>
void MatrixPloop(gauge array, gauge ploop){
  if(array.Type() != SOA || ploop.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || ploop.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  const ArrayType atypein = SOA;
  const ArrayType atypeout = SOA;
  if(PARAMS::UseTex){
	  GAUGE_TEXTURE(array.GetPtr(), true);
    MPloop<true, atypein, atypeout, Real> onepl(array, ploop);
    onepl.Run();
  }
  else{
    MPloop<false, atypein, atypeout, Real> onepl(array, ploop);
    onepl.Run();
  }
}
template void MatrixPloop<float>(gauges array, gauges ploop);
template void MatrixPloop<double>(gauged array, gauged ploop);






















/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////



template<class Real>
struct PLoopArg{
  complex *array;
  complex *value;
};

template <class Real> 
class PLoop: Tunable{
private:
   PLoopArg<Real> arg;
   gauge array;
   int size;
   complex value;
   double timesec;
#ifdef TIMMINGS
    Timer PLooptime;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream);

public:
   PLoop(gauge &array);
   ~PLoop(){dev_free(arg.value);};
   complex Run(const cudaStream_t &stream);
   complex Run();
   double flops();
   double bandwidth();
   long long flop() const ;
   long long bytes() const;
   double time();
   void stat();
   void printValue();
   complex Value()const{return value;}


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


template <int blockSize, bool UseTex, ArrayType atypein, class Real>
__global__ void kernel_calc_polyakovloop(PLoopArg<Real> arg){
	int id = INDEX1D();

  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  complex value = complex::zero();
    
	if(id < DEVPARAMS::tstride){			
		int index = id + 3 * DEVPARAMS::Volume;
		int offset = DEVPARAMS::Volume * 4;

		msun L = GAUGE_LOAD<UseTex, atypein,Real>(arg.array, index, offset);
		for( int t = 1; t < DEVPARAMS::Grid[3]; t++)
			L *= GAUGE_LOAD<UseTex, atypein,Real>(arg.array, index + t * DEVPARAMS::tstride, offset);
			
		value = L.trace();
	}
	complex aggregate = BlockReduce(temp_storage).Reduce(value, Summ<complex>());
	if (threadIdx.x == 0) CudaAtomicAdd(arg.value, aggregate);
}



template <int blockSize, bool UseTex, ArrayType atypein, class Real>
__global__ void kernel_calc_polyakovloopEO(PLoopArg<Real> arg){
	int idd = INDEX1D();

  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  complex value = complex::zero();
    
	if(idd < DEVPARAMS::tstride){		
	
	  int oddbit = 0;
	  int id = idd;
	  if(idd >= DEVPARAMS::tstride / 2){
		  oddbit = 1;
		  id = idd - DEVPARAMS::tstride / 2;
	  }
		  int x[4];
		  Index_4D_EO(x, id, oddbit, DEVPARAMS::Grid);
		  int idx = (((x[2] * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0]);
		  int idl= (x[0] + x[1] + x[2]);
		  int mustride = DEVPARAMS::Volume;
		  int stride = param_Grid(2) * param_Grid(1) * param_Grid(0);
		  int offset = DEVPARAMS::size;

	  msun L = msun::unit();	
	  for( int t = 0; t < DEVPARAMS::Grid[3]; t++){
		  int id0 = (idx + t * stride) >> 1;
		  id0 += ( (idl+t) & 1 ) * param_HalfVolumeG();
		  L *= GAUGE_LOAD<UseTex, atypein,Real>(arg.array, id0 + 3 * mustride, offset);
	  }
		value = L.trace();
	}
	complex aggregate = BlockReduce(temp_storage).Reduce(value, Summ<complex>());
	if (threadIdx.x == 0) CudaAtomicAdd(arg.value, aggregate);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> 
PLoop<Real>::PLoop(gauge &array):array(array){
  //if(array.EvenOdd() == true) errorCULQCD("Not defined for EvenOdd arrays...\n");
	value = complex::zero();
	size = 1;
	for(int i=0;i<3;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;
	arg.array = array.GetPtr();
	arg.value = (complex *)dev_malloc(sizeof(complex));
  
}
template <class Real> 
void PLoop<Real>::apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  CUDA_SAFE_CALL(cudaMemset(arg.value, 0, sizeof(complex)));
  if(array.EvenOdd() == true){
    if(PARAMS::UseTex){
    	//just ensure that the texture was not unbind somewhere...
    	BIND_GAUGE_TEXTURE(array.GetPtr());
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, true, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, true, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, true, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, true, SOA, Real);	
		#endif
	  }
	  else{
		  #if (NCOLORS == 3)
	      if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, false, SOA, Real);		
	      if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, false, SOA12, Real);
	      if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, false, SOA8, Real);
		  #else
	      LAUNCH_KERNEL(kernel_calc_polyakovloopEO, tp, stream, arg, false, SOA, Real);	
		  #endif
	  }
  }
  else{
    if(PARAMS::UseTex){
    	//just ensure that the texture was not unbind somewhere...
    	BIND_GAUGE_TEXTURE(array.GetPtr());
		#if (NCOLORS == 3)
	    if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA, Real);		
	    if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA12, Real);
	    if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA8, Real);
		#else
	    LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, true, SOA, Real);	
		#endif
	  }
	  else{
		  #if (NCOLORS == 3)
	      if(array.Type() == SOA) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA, Real);		
	      if(array.Type() == SOA12) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA12, Real);
	      if(array.Type() == SOA8) LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA8, Real);
		  #else
	      LAUNCH_KERNEL(kernel_calc_polyakovloop, tp, stream, arg, false, SOA, Real);	
		  #endif
	  }
  }
}

template <class Real> 
complex PLoop<Real>::Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    PLooptime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(&value, arg.value, sizeof(complex), cudaMemcpyDeviceToHost));
	  value /= (Real)(NCOLORS * size);

#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    PLooptime.stop();
    timesec = PLooptime.getElapsedTimeInSec();
#endif
	return value;
}
template <class Real> 
complex PLoop<Real>::Run(){	return Run(0);}
template <class Real> 
double PLoop<Real>::time(){	return timesec;}
template <class Real> 
void PLoop<Real>::stat(){	COUT << "PLoop:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
template <class Real> 
void PLoop<Real>::printValue(){	
	printfCULQCD("Polyakov Loop: < %.12e : %.12e : %.12e >\n", value.real(), value.imag(), value.abs());
}
template <class Real> 
long long PLoop<Real>::flop() const {return 0;}
template <class Real> 
long long PLoop<Real>::bytes() const {	return 0;}
template <class Real> 
double PLoop<Real>::flops(){	return ((double)flop() * 1.0e-9) / timesec;}
template <class Real> 
double PLoop<Real>::bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}


template class PLoop<float>;
template class PLoop<double>;


template<class Real>
complex Ploop(gauge array){

  PLoop<Real> onepl(array);
  complex val = onepl.Run();
  onepl.printValue();
  onepl.stat();
  return val;
}
  
template complexs Ploop<float>(gauges array);
template complexd Ploop<double>(gauged array);




} //namespace CULQCD

