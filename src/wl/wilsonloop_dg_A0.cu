
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>



#include <timer.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <constants.h>
#include <matrixsun.h>
#include <gaugearray.h>
#include <index.h>
#include <device_PHB_OVR.h>
#include <reunitlink.h>
#include <staple.h>
#include <comm_mpi.h>
#include <exchange.h>
#include <texture_host.h>

#include <sharedmemtypes.h>

#include <tune.h>
#include <launch_kernel.cuh>


#include <cudaAtomic.h>

#include <cub/cub.cuh>


#include <meas/wloopex.h>


using namespace std;


namespace CULQCD{



template<class Real>
struct WLArgA0{
 const complex *gaugefield;
 const complex *fieldOp;
  Real *wloop;
  int radius;
  int Tmax;
  int mu;
  int opN;
};




#if __CUDA_ARCH__ >= 350
// Device has ldg
template<typename T>
__device__ __forceinline__ T Aldg(const T* ptr) {
    return __ldg(ptr);
}
#else
//Device does not, fall back.
template<typename T>
__device__ __forceinline__ T Aldg(const T* ptr) {
    return *ptr;
}
#endif


template<int blockSize, bool UseTex, class Real, ArrayType atype>
__global__ void kernel_WilsonLoop_A0(WLArgA0<Real> arg){

  typedef cub::BlockReduce<Real, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;  

  int id = INDEX1D();
	int x[4];
	Index_4D_NM(id, x);

	int idl = ( x[2] * param_Grid(1) + x[1] ) * param_Grid(0) + x[0]; //space index left
	x[arg.mu] = (x[arg.mu]+arg.radius)%param_Grid(arg.mu);
	int idr = ( x[2] * param_Grid(1) + x[1] ) * param_Grid(0) + x[0]; //space index right

	int tdirvolume = 3 * DEVPARAMS::Volume;
	int gfoffset = arg.opN * DEVPARAMS::Volume;


	msun t0 = msun::identity();
	msun t1 = msun::identity();
	for(int it = 0; it <= arg.Tmax; it++){
		int idt = (x[3]+it)%param_Grid(3);
		idt *= DEVPARAMS::tstride;
        int idOp = 0;
          for(int iop = 0; iop < arg.opN; iop++)
          for(int jop = 0; jop < arg.opN; jop++){
            msun linkb = msun::zero();
            if(id < DEVPARAMS::Volume) linkb = GAUGE_LOAD<false, atype, Real>( arg.fieldOp, id + iop * DEVPARAMS::Volume, gfoffset); //bottom space links
            Real w = 0.0;
		        if(id < DEVPARAMS::Volume){
			        msun linkt = GAUGE_LOAD_DAGGER<false, atype, Real>( arg.fieldOp, idl + idt + jop * DEVPARAMS::Volume, gfoffset); //top space links
			        w = (linkb * t1 * linkt * t0.dagger()).realtrace();
		        }
                int wloffset = it + idOp * (arg.Tmax+1); 
				Real aggregate = BlockReduce(temp_storage).Reduce(w, Summ<Real>());
				if (threadIdx.x == 0) CudaAtomicAdd(arg.wloop + wloffset, aggregate);
				__syncthreads();
                idOp++;            
          } 
		  if(id < DEVPARAMS::Volume && it < arg.Tmax){
			  t0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idl + idt + tdirvolume, DEVPARAMS::size);
			  t1 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idr + idt + tdirvolume, DEVPARAMS::size);
		  } 
    }
}











template <bool UseTex, class Real, ArrayType atype> 
class WilsonLoop_A0: Tunable{
private:
   WLArgA0<Real> arg;
   gauge array;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer WilsonLoop_A0time;
#endif
	TuneParam tp;
	Real *wloop_tmp;
    size_t wloop_mem;
	Real *field_tmp;
    size_t field_mem;

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	tp = tuneLaunch(*this, getTuning(), getVerbosity());
  LAUNCH_KERNEL(kernel_WilsonLoop_A0, tp, stream, arg, UseTex, Real, atype);
}
public:
   WilsonLoop_A0(WLArgA0<Real> arg, gauge array): arg(arg), array(array){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
  void SetIop(int iop){ arg.iop = iop;}
   ~WilsonLoop_A0(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    WilsonLoop_A0time.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    WilsonLoop_A0time.stop();
    timesec = WilsonLoop_A0time.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double time(){	return timesec;}
   void stat(){	COUT << "WilsonLoop_A0:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { 
    wloop_mem = arg.opN*arg.opN * (arg.Tmax+1) * sizeof(Real);
	wloop_tmp = (Real*) safe_malloc( wloop_mem );
  	CUDA_SAFE_CALL(cudaMemcpy(wloop_tmp, arg.wloop, wloop_mem, cudaMemcpyDeviceToHost));	
  }
  void postTune() {  
  	CUDA_SAFE_CALL(cudaMemcpy(arg.wloop, wloop_tmp, wloop_mem, cudaMemcpyHostToDevice));	
	host_free(wloop_tmp);
  }

};








template<bool UseTex, class Real>
void CalcWilsonLoop_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu){
  Timer mtime;
  mtime.start(); 
  WLArgA0<Real> argK;
	argK.gaugefield = array.GetPtr();
	argK.fieldOp = arg->fieldOp.GetPtr();
	argK.wloop = arg->wloop;
	argK.radius = radius;
	argK.Tmax = arg->Tmax;
	argK.mu = mu;
	argK.opN = arg->opN;	

  
  if(array.Type() != SOA || arg->fieldOp.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || arg->fieldOp.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  WilsonLoop_A0<UseTex, Real, SOA> wl(argK, array);
  wl.Run();
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
 if (getVerbosity() >= VERBOSE) COUT << "Time WilsonLoop_A0F:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}



template<class Real>
void CalcWilsonLoop_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalcWilsonLoop_A0<true, Real>(array, arg, radius, mu);
  }
  else CalcWilsonLoop_A0<false, Real>(array, arg, radius, mu);
}


template void CalcWilsonLoop_A0<double>(gauged array, Sigma_g_plus<double> *arg, int radius, int mu);



}

