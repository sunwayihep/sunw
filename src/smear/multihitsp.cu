
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


#include <tune.h>


using namespace std;


namespace CULQCD{



template <bool UseTex, ArrayType atype, class Real> 
__device__ void inline 
CalcStapleSP(
	complex *array, 
	msun &staple, 
	int idx, 
	int mu
){
  int x[NDIMS];
  Index_ND_NM(idx, x);
  int mustride = DEVPARAMS::Volume;
  int muvolume = mu * mustride;
  int offset = DEVPARAMS::size;
	//int newidmu1 = Index_4D_Neig_NM(idx, mu, 1);
	for(int nu = 0; nu < NDIMS-1; nu++)  if(mu != nu) {
    int dx[NDIMS] = {0};
		int nuvolume = nu * mustride;
		msun link;	
		//UP
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idx + nuvolume, offset);
		dx[nu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx) + muvolume, offset);	
		dx[nu]--;
		dx[mu]++;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx) + nuvolume, offset);
		staple += link;
		dx[mu]--;
    //DOWN
		dx[nu]--;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_ND_Neig_NM(x,dx) + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx)  + muvolume, offset);
		dx[mu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx) + nuvolume, offset);
		staple += link;
	}
}

template <bool UseTex, ArrayType atype, class Real> 
__device__ void inline 
CalcStaple(
	complex *array, 
	msun &staple, 
	int idx, 
	int mu
){
  int x[NDIMS];
  Index_ND_NM(idx, x);
  int mustride = DEVPARAMS::Volume;
  int muvolume = mu * mustride;
  int offset = DEVPARAMS::size;
	//int newidmu1 = Index_4D_Neig_NM(idx, mu, 1);
	for(int nu = 0; nu < NDIMS; nu++)  if(mu != nu) {
    int dx[NDIMS] = {0};
		int nuvolume = nu * mustride;
		msun link;	
		//UP
		link = GAUGE_LOAD<UseTex, atype, Real>( array,  idx + nuvolume, offset);
		dx[nu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx) + muvolume, offset);	
		dx[nu]--;
		dx[mu]++;
		link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx) + nuvolume, offset);
		staple += link;
		dx[mu]--;
    //DOWN
		dx[nu]--;
		link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( array,  Index_ND_Neig_NM(x,dx) + nuvolume, offset);	
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx)  + muvolume, offset);
		dx[mu]++;
		link *= GAUGE_LOAD<UseTex, atype, Real>( array, Index_ND_Neig_NM(x,dx) + nuvolume, offset);
		staple += link;
	}
}






//kernel MultiHitSP, uses random array state with volume/2 size
template <bool UseTex, ArrayType atypeIn, ArrayType atypeOut, class Real> 
__global__ void 
kernel_MultiHitSP_1D_halfrng(
		complex *arrayin,
		complex *arrayout,
		cuRNGState *state, 
		int mmu,
		int nhit
		){
	int idd = INDEX1D();
	//if(idd >= DEVPARAMS::Volume) return;
	if(idd >= DEVPARAMS::HalfVolume) return;	
	cuRNGState localState = state[ idd ];
	for(int lat = 0; lat < 2; lat++){	
		int id = idd + lat * DEVPARAMS::HalfVolume;
		for(int mu=0;mu<NDIMS-1;mu++){
		  msun staple = msun::zero();
		  //calculate the staple
		  CalcStapleSP<UseTex, atypeIn, Real>(arrayin, staple, id, mu);
		  msun U = GAUGE_LOAD<UseTex, atypeIn, Real>( arrayin, id + mu * DEVPARAMS::Volume);
		  staple = staple.dagger();
		  msun link = msun::zero();
		  for(int iter = 0; iter < nhit; iter++){
			  //heatBathSUN<Real>( U, staple, localState );
			  overrelaxationSUN<Real>( U, staple );
			  link += U;
		  }
		  link /= (Real)(nhit);	
		  GAUGE_SAVE<atypeOut,Real>( arrayout, link, id + mu * DEVPARAMS::Volume );
		}
		if(0){
		  int mu = 3;
		  msun staple = msun::zero();
		  //calculate the staple
		  CalcStaple<UseTex, atypeIn, Real>(arrayin, staple, id, mu);
		  msun U = GAUGE_LOAD<UseTex, atypeIn, Real>( arrayin, id + mu * DEVPARAMS::Volume);
		  staple = staple.dagger();
		  msun link = msun::zero();
		  for(int iter = 0; iter < nhit; iter++){
			  //heatBathSUN<Real>( U, staple, localState );
			  overrelaxationSUN<Real>( U, staple );
			  link += U;
		  }
		  link /= (Real)(nhit);	
		  GAUGE_SAVE<atypeOut,Real>( arrayout, link, id + mu * DEVPARAMS::Volume );
		}
	}
	state[ idd ] = localState;
}





template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real> 
class MultiHitSP: Tunable{
private:
   gauge arrayin;
   gauge arrayout;
   RNG &randstates;
   int size;
   double timesec;
   int mu;
   int nhit;
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
      kernel_MultiHitSP_1D_halfrng<UseTex, atypein, atypeout, Real><<<tp.grid,tp.block, 0, stream>>>(arrayin.GetPtr(), arrayout.GetPtr(), randstates.State(), mu, nhit);
	}
public:
   MultiHitSP(gauge &arrayin, gauge &arrayout, RNG &randstates, int mu, int nhit):arrayin(arrayin), arrayout(arrayout), randstates(randstates), mu(mu), nhit(nhit){
		size = 1;
		//Number of threads is equal to the number of space points!
		for(int i=0;i<NDIMS;i++){
		  size *= PARAMS::Grid[i];
		} 
		size = size /2 ;
		timesec = 0.0;
	}
  ~MultiHitSP(){};

	double time(){return timesec;}
	void stat(){ COUT << "MultiHitSP:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
	long long flop() const { return 0;}
	long long bytes() const { return 0;}
	double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
	double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
	for(int i=0; i<NDIMS-1; i++) vol << PARAMS::Grid[i] << "x";
    vol << PARAMS::Grid[NDIMS-1];
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
	void preTune() { arrayin.Backup(); randstates.Backup(); }
	void postTune() { arrayin.Restore(); randstates.Restore(); }
	
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
void ApplyMultiHitSpace(gauge array, gauge arrayout, RNG &randstates, int nhit){
  
  if(array.Type() != SOA || arrayout.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || arrayout.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  const ArrayType atypein = SOA;
  const ArrayType atypeout = SOA;
  if(PARAMS::UseTex){
	  GAUGE_TEXTURE(array.GetPtr(), true);
    MultiHitSP<true, atypein, atypeout, Real> mhit(array, arrayout, randstates, NDIMS-1, nhit);
    mhit.Run();
    mhit.stat();
  }
  else{
    MultiHitSP<false, atypein, atypeout, Real> mhit(array, arrayout, randstates, NDIMS-1, nhit);
    mhit.Run();
    mhit.stat();
  }
}
template void ApplyMultiHitSpace<float>(gauges array, gauges arrayout, RNG &randstates, int nhit);
template void ApplyMultiHitSpace<double>(gauged array, gauged arrayout, RNG &randstates, int nhit);





}
