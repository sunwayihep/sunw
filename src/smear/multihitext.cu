
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>



#include <timer.h>
#include <cuda_common.h>
#include <monte/monte.h>
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




//kernel MultiHitExt, uses random array state with volume/2 size
template <bool UseTex, ArrayType atypeIn, ArrayType atypeOut, class Real> 
__global__ void 
kernel_multihitext_1D_halfrng111(
    complex *arrayin,
    complex *arrayout,
    cuRNGState *state, 
    int mu,
    int nhit
    ){
  uint idd = INDEX1D();
  if(idd >= DEVPARAMS::HalfVolume) return;  
  cuRNGState localState = state[ idd ];
  for(int iter = 0; iter < 2; iter++){
    int id = idd + iter * DEVPARAMS::HalfVolume;
    msun staple = msun::zero();
    //calculate the staple
    Staple<UseTex, atypeIn, Real>(arrayin, staple, id, mu);
    msun U = GAUGE_LOAD<UseTex, atypeIn, Real>( arrayin, id + mu * DEVPARAMS::Volume);
    staple = staple.dagger();
    msun link = U;
    for(int iter = 0; iter < nhit; iter++){
      heatBathSUN<Real>( U, staple, localState );
      link +=U;
    }
    link /= (Real)(nhit+1); 
    GAUGE_SAVE<atypeOut,Real>( arrayout, link, id + mu * DEVPARAMS::Volume );
  }
  state[ idd ] = localState;
}


struct multihitArg{
  int b[4];
  //extended lattice dim
  int grid[4];
  //extend lattice volume
  int volume;
  //extend lattice offset
  int offset;
  //number of sublattices
  int blocogrid[4];
  //sublattice dim
  int lgrid[4];
  int id;
  int len[4];
  //active threas = num sublattices
  int totblocks;
  int intids;
  
  void init(int lenx, int lent){
    for(int d=0; d<3; d++) len[d] = lenx;
    len[3] = lent;
    for(int d=0; d<4; d++) lgrid[d] = 2*len[d];
    for(int d=0; d<4; d++) blocogrid[d] = (PARAMS::Grid[d] + lgrid[d] -1) / lgrid[d];
    for(int d=0; d<4; d++) b[d] = 2; 
    //b[3]=1;
    for(int d=0; d<4; d++) grid[d] = blocogrid[d] * lgrid[d] + 2 * b[d];
    volume = 1;
    for(int d=0; d<4; d++) volume *= grid[d];
    offset = 4 * volume; 
    totblocks = 1;
    for(int d=0; d<4; d++) totblocks *= blocogrid[d];
    intids = 1;
    for(int d=0; d<4; d++) intids *= lgrid[d];
   
COUT << "PARAMS::Grid:  " <<  PARAMS::Grid[0] << ":" << PARAMS::Grid[1] << ":" << PARAMS::Grid[2] << ":" << PARAMS::Grid[3] << endl;
COUT << "len:  " <<  len[0] << ":" << len[1] << ":" << len[2] << ":" << len[3] << endl;
COUT << "lgrid:  " <<  lgrid[0] << ":" << lgrid[1] << ":" << lgrid[2] << ":" << lgrid[3] << endl;
COUT << "blocogrid:  " <<  blocogrid[0] << ":" << blocogrid[1] << ":" << blocogrid[2] << ":" << blocogrid[3] << endl;
COUT << "b:  " <<  b[0] << ":" << b[1] << ":" << b[2] << ":" << b[3] << endl;
COUT << "grid:  " <<  grid[0] << ":" << grid[1] << ":" << grid[2] << ":" << grid[3] << endl;
COUT << "volume:  " <<  volume  << endl;
COUT << "offset:  " <<  offset  << endl;
COUT << "totblocks:  " <<  totblocks  << endl;
COUT << "intids:  " <<  intids  << endl;
COUT << "id:  " <<  id  << endl;
  }

};


inline  __host__   __device__ void get4DFLL(int id, int x[4], int X[4]){
  x[3] = id/(X[0] * X[1] * X[2]);
  x[2] = (id/(X[0] * X[1])) % X[2];
  x[1] = (id/X[0]) % X[1];
  x[0] = id % X[0]; 
}

template <bool UseTex, ArrayType atypeIn, ArrayType atypeOut, class Real> 
__global__ void 
kernel_make_extendedLattice(
    complex *arrayin,
    complex *arrayout,
    multihitArg arg
    ){
  uint id = INDEX1D();
  if(id >= arg.volume) return;  
  int xext[4];
  get4DFLL(id, xext, arg.grid);
  int x[4];
  for(int i = 0; i < 4; i++)
    x[i] = (xext[i] - arg.b[i] + param_Grid(i) ) % param_Grid(i);
  int idin = (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0];
  int idout = (((xext[3] * arg.grid[2] + xext[2]) * arg.grid[1]) + xext[1] ) * arg.grid[0] + xext[0];
  for(int mu = 0; mu < 4; mu++){
    msun U = GAUGE_LOAD<UseTex, atypeIn, Real>( arrayin, idin + mu * DEVPARAMS::Volume);
    GAUGE_SAVE<atypeOut,Real>( arrayout, U, idout + mu * arg.volume, arg.offset );
  }
} 





template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real> 
class MultiHitExtCopy: Tunable{
private:
   gauge arrayin;
   gauge arrayout;
   int size;
   double timesec;
   int mu;
   int nhit;
   multihitArg arg;
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
      kernel_make_extendedLattice<UseTex, atypein, atypeout, Real><<<tp.grid,tp.block, 0, stream>>>(arrayin.GetPtr(), arrayout.GetPtr(), arg);
  }
public:
   MultiHitExtCopy(gauge &arrayin, gauge &arrayout, multihitArg &arg):arrayin(arrayin), arrayout(arrayout), arg(arg){
    size = arg.volume;
    timesec = 0.0;
  }
  ~MultiHitExtCopy(){};

  double time(){return timesec;}
  void stat(){ COUT << "MultiHitExtCopy:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  long long flop() const { return 0;}
  long long bytes() const { return 0;}
  double flops(){ return ((double)flop() * 1.0e-9) / timesec;}
  double bandwidth(){ return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << arg.grid[0] << "x";
    vol << arg.grid[1] << "x";
    vol << arg.grid[2] << "x";
    vol << arg.grid[3];
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
  void postTune() { }
  
  void Run(const cudaStream_t &stream){
  #ifdef TIMMINGS
      mtime.start();
  #endif
      if(UseTex){
        BIND_GAUGE_TEXTURE(arrayin.GetPtr());
      }
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





__device__ __host__ inline int linkIndex22(int x[], int dx[], int grid[]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + grid[i]) % grid[i];
  return (((y[3]*grid[2] + y[2])*grid[1] + y[1])*grid[0] + y[0]);
}         
          
template <bool UseTex, ArrayType atypeIn, ArrayType atypeOut, class Real> 
__global__ void 
kernel_multihitext(
    complex *array,
    complex *arrayout,
    cuRNGState *state, 
    int mu,
    int nhit,
    multihitArg arg
    ){
  int idd = INDEX1D(); //lattice block id
  if(idd >= arg.totblocks) return;  
  //get sublattice block id
  int xb[4];
  get4DFLL(idd, xb, arg.blocogrid);
  
  //get current sublatice id to update
  int lid[4];
  get4DFLL(arg.id, lid, arg.lgrid);
  
  //get lattice pos for current
  int x[4];
  for(int d=0; d < 4; d++) x[d] = lid[d] + xb[d] * arg.lgrid[d];
    
  //normal lattice if to save final result
  if( x[0] >= param_Grid(0) || x[1] >= param_Grid(1) || x[2] >= param_Grid(2) || x[3] >= param_Grid(3) )
    return;
  //current id on the final update gauge
  int realid = (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0];
  
  //add lattice offsets  
  for(int d=0; d < 4; d++) x[d] += arg.b[d];
  //current id in extended lattice
  int mainid = (((x[3] * arg.grid[2] + x[2]) * arg.grid[1]) + x[1] ) * arg.grid[0] + x[0];
  
  cuRNGState localState = state[ idd ];
  //Start calculating
  msun newU = msun::zero();
  for( int iter = 0; iter < nhit; ++iter ) {
    for( int parity = 0; parity < 2; ++parity ) {
  
    // TODO: CHECK THIS RANGES!!!!!!!!
    for( int l = -arg.len[3] + 1; l < arg.len[3]; ++l )
    for( int k = -arg.len[2] + 1; k < arg.len[2]; ++k )
    for( int j = -arg.len[1] + 1; j < arg.len[1]; ++j )
    for( int i = -arg.len[0] + 1; i < arg.len[0]; ++i ) {
      if ( ((i+j+k + l)&1) != parity )
      //if ( ((i+j+k)&1) != parity )
        continue;
          int xx[4];
          xx[0] = i + x[0];
          xx[1] = j + x[1];
          xx[2] = k + x[2];
          xx[3] = l + x[3];
          //xx[3] = x[3];
          int muvolume = mu * arg.volume;
          int idx = (((xx[3] * arg.grid[2] + xx[2]) * arg.grid[1]) + xx[1] ) * arg.grid[0] + xx[0];
          msun staple = msun::zero();
          for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
            int nuvolume = nu * arg.volume;
            msun link;  
            int dx[4] = {0, 0, 0, 0};
            //UP
            link = GAUGE_LOAD<UseTex, atypeIn, Real>( array,  idx + nuvolume, arg.offset);
            dx[nu]++;
            link *= GAUGE_LOAD<UseTex, atypeIn, Real>( array, linkIndex22(xx,dx, arg.grid) + muvolume, arg.offset); 
            dx[nu]--;
            dx[mu]++;
            link *= GAUGE_LOAD_DAGGER<UseTex, atypeIn, Real>( array, linkIndex22(xx,dx, arg.grid) + nuvolume, arg.offset);
            staple += link;
            dx[mu]--;
            //DOWN
            dx[nu]--;
            link = GAUGE_LOAD_DAGGER<UseTex, atypeIn, Real>( array,  linkIndex22(xx,dx, arg.grid) + nuvolume, arg.offset);  
            link *= GAUGE_LOAD<UseTex, atypeIn, Real>( array, linkIndex22(xx,dx, arg.grid)  + muvolume, arg.offset);
            dx[mu]++;
            link *= GAUGE_LOAD<UseTex, atypeIn, Real>( array, linkIndex22(xx,dx, arg.grid) + nuvolume, arg.offset);
            staple += link;
          }
		    staple = staple.dagger();
        msun U = GAUGE_LOAD<UseTex, atypeIn, Real>( array,  idx + muvolume, arg.offset);          
        heatBathSUN<Real>( U, staple, localState );
        GAUGE_SAVE<atypeIn,Real>( array, U, idx + muvolume, arg.offset );
      }
    }
    newU += GAUGE_LOAD<UseTex, atypeIn, Real>( array,  mainid + mu * arg.volume, arg.offset);
  }
  newU /= (Real)nhit;
  GAUGE_SAVE<atypeOut,Real>( arrayout, newU, realid + mu * DEVPARAMS::Volume );
  state[ idd ] = localState;
} 




template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real> 
class MultiHitExt: Tunable{
private:
   gauge arrayin;
   gauge arrayout;
   RNG &randstates;
   int size;
   double timesec;
   int mu;
   int nhit;
   multihitArg arg;
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
      kernel_multihitext<UseTex, atypein, atypeout, Real><<<tp.grid,tp.block, 0, stream>>>(arrayin.GetPtr(), arrayout.GetPtr(), randstates.State(), mu, nhit, arg);
  }
public:
   MultiHitExt(gauge &arrayin, gauge &arrayout, RNG &randstates, int mu, int nhit, multihitArg arg):arrayin(arrayin), arrayout(arrayout), randstates(randstates), mu(mu), nhit(nhit), arg(arg){
    size = arg.totblocks ;
COUT << "arg.totblocks:  " <<  arg.totblocks  << endl;
    timesec = 0.0;
  }
  ~MultiHitExt(){};

  double time(){return timesec;}
  void stat(){ COUT << "MultiHitExt:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  long long flop() const { return 0;}
  long long bytes() const { return 0;}
  double flops(){ return ((double)flop() * 1.0e-9) / timesec;}
  double bandwidth(){ return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << arg.blocogrid[0] << "x";
    vol << arg.blocogrid[1] << "x";
    vol << arg.blocogrid[2] << "x";
    vol << arg.blocogrid[3];
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
  
  void Run(const cudaStream_t &stream, int id){
  #ifdef TIMMINGS
      mtime.start();
  #endif
    if(UseTex){
      BIND_GAUGE_TEXTURE(arrayin.GetPtr());
    }
      arg.id = id;
      apply(stream);
  #ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      mtime.stop();
      timesec = mtime.getElapsedTimeInSec();
  #endif
  }
  //void Run(int id){return Run(0, id);}
};



#define LEN_SPACE 2
#define LEN_TIME 1


template<bool UseTex, class Real>
void ApplyMultiHitExtended(gauge array, gauge arrayout, RNG &randstates, int mu, int nhit){
  if(array.Type() != SOA || arrayout.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || arrayout.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  const ArrayType atypein = SOA;
  const ArrayType atypeout = SOA;
  
  COUT << "MultiHitExtended: nhit = " <<  nhit  << endl;
  Timer mtime;
  mtime.start();
    multihitArg arg;
    arg.init(LEN_SPACE,LEN_TIME);
    if(randstates.Size() < arg.totblocks) errorCULQCD("Size of the RNG is small...\n");
    gauge CC(array.Type(), Device, arg.offset, false);  
    MultiHitExtCopy<UseTex, atypein, atypeout, Real> chitCopy(array, CC, arg);
    MultiHitExt<UseTex, atypein, atypeout, Real> mhit(CC, arrayout, randstates, mu, nhit, arg);
    for(int id = 0; id < arg.intids; id++){
      chitCopy.Run();
      mhit.Run(0, id);
    }
    CC.Release();
    chitCopy.stat();
    mhit.stat();
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  COUT << "Time MultiHitExtended:  " <<  mtime.getElapsedTimeInSec()  << " s"  << endl;
}
#undef LEN_SPACE
#undef LEN_TIME


template<class Real>
void ApplyMultiHitExt(gauge array, gauge arrayout, RNG &randstates, int nhit){
  if(PARAMS::UseTex){
    ApplyMultiHitExtended<true, Real>(array, arrayout, randstates, 3, nhit);
  }
  else{
    ApplyMultiHitExtended<false, Real>(array, arrayout, randstates, 3, nhit);
  }
}
template void ApplyMultiHitExt<float>(gauges array, gauges arrayout, RNG &randstates, int nhit);
template void ApplyMultiHitExt<double>(gauged array, gauged arrayout, RNG &randstates, int nhit);





}
