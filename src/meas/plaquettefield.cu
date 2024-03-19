

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>


#include <device_load_save.h>
#include <cuda_common.h>
#include <constants.h>
#include <index.h>
#include <reduction.h>
#include <timer.h>
#include <texture_host.h>
#include <comm_mpi.h>


#include <tune.h>
#include <launch_kernel.cuh>
#include <cudaAtomic.h>

#include <cub/cub.cuh>


using namespace std;


namespace CULQCD{









template<class Real>
struct PlaquettedArg{
  complex *array;
  complex *plaq;
  complex *meanplaq;
};



template<bool UseTex, ArrayType atype, class Real>
inline   __device__ complex Plaquette(const complex *array, const int idx, const int mu, const int nu){
  int x[4];
  Index_4D_NM(idx, x);
  int mustride = DEVPARAMS::Volume;
  int muvolume = mu * mustride;
	int nuvolume = nu * mustride;
  int offset = DEVPARAMS::size;
  int dx[4] = {0, 0, 0, 0};
  msun link = GAUGE_LOAD<UseTex, atype, Real>( array, idx + muvolume, offset);
  dx[mu]++;
	link *= GAUGE_LOAD<UseTex, atype, Real>( array,  Index_4D_Neig_NM(x,dx) + nuvolume, offset);
  dx[mu]--;
	dx[nu]++;
	link *= GAUGE_LOAD_DAGGER<UseTex, atype,Real>( array, Index_4D_Neig_NM(x,dx) + muvolume, offset);		
	dx[nu]--;	
	link *= GAUGE_LOAD_DAGGER<UseTex, atype,Real>( array, idx + nuvolume, offset);

  //return -link.trace() / 3.0 + 1.0;
  return (link.trace() / 3.0);
}  





template<bool UseTex, ArrayType atypein, class Real>
inline   __device__ void SixPlaquette(const complex *array, complex *plaq, const int idx){
  plaq[0] += Plaquette<UseTex, atypein, Real>( array, idx, 0, 3 );
  plaq[1] += Plaquette<UseTex, atypein, Real>( array, idx, 1, 3 );
  plaq[2] += Plaquette<UseTex, atypein, Real>( array, idx, 2, 3 );
  plaq[3] += Plaquette<UseTex, atypein, Real>( array, idx, 1, 2 );
  plaq[4] += Plaquette<UseTex, atypein, Real>( array, idx, 2, 0 );
  plaq[5] += Plaquette<UseTex, atypein, Real>( array, idx, 0, 1 );
}








template <int blockSize, bool UseTex, ArrayType atypein, class Real, bool spacetime>
__global__ void kernel_calc_plaquette(PlaquettedArg<Real> arg){ 
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;  
        
	int id = INDEX1D();

  complex plaq[6];
  for(int d=0; d<6; d++) plaq[d] = complex::zero();
  if( spacetime ) {
	  if(id < DEVPARAMS::Volume) {
		  SixPlaquette<UseTex, atypein, Real>(arg.array, plaq, id);
	  }
	  for(int d=0; d<6; d++) arg.plaq[id + d * DEVPARAMS::Volume] = plaq[d];
  }
  else {
	  if(id < DEVPARAMS::tstride) {
		for(int it = 0; it < DEVPARAMS::Grid[3]; it++) {
		  SixPlaquette<UseTex, atypein, Real>(arg.array, plaq, id + it * DEVPARAMS::tstride);
		}
		for(int d=0; d<6; d++) plaq[d] /= (Real)DEVPARAMS::Grid[3];
		for(int d=0; d<6; d++) arg.plaq[id + d * DEVPARAMS::tstride] = plaq[d];
	  }
  }
  for(int d=0; d<6; d++){
      complex aggregate = BlockReduce(temp_storage).Reduce(plaq[d], Summ<complex>());
      if(threadIdx.x==0) CudaAtomicAdd(arg.meanplaq + d, aggregate);
      __syncthreads();
  }
}
  
  
  
  





template <bool UseTex, ArrayType atypein, class Real, bool spacetime> 
class PlaqField: Tunable{
private:
   gauge arrayin;
   PlaquettedArg<Real> arg;
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
      //kernel_calc_plaquette<UseTex, atypein, Real><<<tp.grid,tp.block, 0, stream>>>(arg);
      CUDA_SAFE_CALL(cudaMemset(arg.meanplaq, 0, 6 * sizeof(complex)));
      LAUNCH_KERNEL(kernel_calc_plaquette, tp, stream, arg, UseTex, atypein, Real, spacetime);
	}
public:
   PlaqField(gauge &arrayin, complex *plaq, complex *meanplaq):arrayin(arrayin){
		size = 1;
		//Number of threads is equal to the number of space points!
		for(int i=0;i<3;i++){
		  size *= PARAMS::Grid[i];
		} 
		if( spacetime ) size *= PARAMS::Grid[3];
		size = size;
		timesec = 0.0;
	  arg.array = arrayin.GetPtr();
	  arg.plaq = plaq;
	  arg.meanplaq = meanplaq;
	}
  ~PlaqField(){};

	double time(){return timesec;}
	void stat(){ COUT << "PlaqField:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
	void postTune() {}
	
	void Run(const cudaStream_t &stream){
	#ifdef TIMMINGS
	    mtime.start();
	#endif
	    apply(stream);    
	#ifdef TIMMINGS
		CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Kernel execution failed");
	    mtime.stop();
	    timesec = mtime.getElapsedTimeInSec();
	#endif
	}
	void Run(){return Run(0);}
};




template<class Real>
void PlaquetteFieldSpace(gauge array, complex *plaq, complex *meanplaq){
  
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  const ArrayType atypein = SOA;
  if(PARAMS::UseTex){
	  GAUGE_TEXTURE(array.GetPtr(), true);
    PlaqField<true, atypein, Real, false> pf(array, plaq, meanplaq);
    pf.Run();
    pf.stat();
  }
  else{
    PlaqField<false, atypein, Real, false> pf(array, plaq, meanplaq);
    pf.Run();
    pf.stat();
  }
}
template void PlaquetteFieldSpace<float>(gauges array, complexs *plaq, complexs *meanplaq);
template void PlaquetteFieldSpace<double>(gauged array, complexd *plaq, complexd *meanplaq);





template<class Real>
void PlaquetteField(gauge array, complex *plaq, complex *meanplaq){
  
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  const ArrayType atypein = SOA;
  if(PARAMS::UseTex){
	  GAUGE_TEXTURE(array.GetPtr(), true);
    PlaqField<true, atypein, Real, true> pf(array, plaq, meanplaq);
    pf.Run();
    pf.stat();
  }
  else{
    PlaqField<false, atypein, Real, true> pf(array, plaq, meanplaq);
    pf.Run();
    pf.stat();
  }
}
template void PlaquetteField<float>(gauges array, complexs *plaq, complexs *meanplaq);
template void PlaquetteField<double>(gauged array, complexd *plaq, complexd *meanplaq);






}
