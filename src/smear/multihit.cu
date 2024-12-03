
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <comm_mpi.h>
#include <constants.h>
#include <cuda_common.h>
#include <device_PHB_OVR.h>
#include <device_load_save.h>
#include <exchange.h>
#include <gaugearray.h>
#include <index.h>
#include <matrixsun.h>
#include <reunitlink.h>
#include <staple.h>
#include <texture_host.h>
#include <timer.h>

#include <tune.h>

using namespace std;

namespace CULQCD {

template <bool UseTex, ArrayType atype, class Real>
__device__ void inline CalcStaple(complex *array, msun &staple, int idx,
                                  int mu) {
  int x[NDIMS];
  Index_ND_NM(idx, x);
  int mustride = DEVPARAMS::Volume;
  int muvolume = mu * mustride;
  int offset = DEVPARAMS::size;
  // int newidmu1 = Index_4D_Neig_NM(idx, mu, 1);
  for (int nu = 0; nu < NDIMS; nu++)
    if (mu != nu) {
      int dx[NDIMS] = {0};
      int nuvolume = nu * mustride;
      msun link;
      // UP
      link = GAUGE_LOAD<UseTex, atype, Real>(array, idx + nuvolume, offset);
      dx[nu]++;
      link *= GAUGE_LOAD<UseTex, atype, Real>(
          array, Index_ND_Neig_NM(x, dx) + muvolume, offset);
      dx[nu]--;
      dx[mu]++;
      link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
          array, Index_ND_Neig_NM(x, dx) + nuvolume, offset);
      staple += link;
      dx[mu]--;
      // DOWN
      dx[nu]--;
      link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
          array, Index_ND_Neig_NM(x, dx) + nuvolume, offset);
      link *= GAUGE_LOAD<UseTex, atype, Real>(
          array, Index_ND_Neig_NM(x, dx) + muvolume, offset);
      dx[mu]++;
      link *= GAUGE_LOAD<UseTex, atype, Real>(
          array, Index_ND_Neig_NM(x, dx) + nuvolume, offset);
      staple += link;
    }
}

// kernel MultiHit, uses random array state with volume/2 size
template <bool UseTex, ArrayType atypeIn, ArrayType atypeOut, class Real>
__global__ void kernel_multihit_1D_halfrng(complex *arrayin, complex *arrayout,
                                           cuRNGState *state, int mu,
                                           int nhit) {
  int idd = INDEX1D();
  if (idd >= DEVPARAMS::HalfVolume)
    return;
  cuRNGState localState = state[idd];
  for (int lat = 0; lat < 2; lat++) {
    int id = idd + lat * DEVPARAMS::HalfVolume;
    msun staple = msun::zero();
    // calculate the staple
    CalcStaple<UseTex, atypeIn, Real>(arrayin, staple, id, mu);
    msun U =
        GAUGE_LOAD<UseTex, atypeIn, Real>(arrayin, id + mu * DEVPARAMS::Volume);
    staple = staple.dagger();
    msun link = U;
    for (int iter = 0; iter < nhit; iter++) {
      heatBathSUN<Real>(U, staple, localState);
      link += U;
    }
    link /= (Real)(nhit + 1);
    GAUGE_SAVE<atypeOut, Real>(arrayout, link, id + mu * DEVPARAMS::Volume);
  }
  state[idd] = localState;
}

template <bool UseTex, ArrayType atype, ArrayType atypehit, class Real>
__global__ void kernel_evenodd(complex *arrayin, complex *arrayout,
                               cuRNGState *state, int mu, int oddbit,
                               int nhit) {
  int id = INDEX1D();
  if (id >= DEVPARAMS::HalfVolume)
    return;
  int mustride = DEVPARAMS::Volume;
  int muvolume = mu * mustride;
  int offset = DEVPARAMS::size;
  int idxoddbit = id + oddbit * param_HalfVolume();

  msun staple = msu3::zero();
  int newidmu1 = Index_ND_Neig_EO(id, oddbit, mu, 1);
  for (int nu = 0; nu < NDIMS; nu++) {
    if (mu == nu)
      continue;
    msun link;
    int nuvolume = nu * mustride;
    // UP
    link =
        GAUGE_LOAD<UseTex, atype, Real>(arrayin, idxoddbit + nuvolume, offset);
    link *= GAUGE_LOAD<UseTex, atype, Real>(
        arrayin, Index_ND_Neig_EO(id, oddbit, nu, 1) + muvolume, offset);
    link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arrayin, newidmu1 + nuvolume,
                                                   offset);
    staple += link;
    // DOWN
    int newidnum1 = Index_ND_Neig_EO(id, oddbit, nu, -1);
    link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arrayin, newidnum1 + nuvolume,
                                                  offset);
    link *=
        GAUGE_LOAD<UseTex, atype, Real>(arrayin, newidnum1 + muvolume, offset);
    link *= GAUGE_LOAD<UseTex, atype, Real>(
        arrayin, Index_ND_Neig_EO(id, oddbit, mu, 1, nu, -1) + nuvolume,
        offset);
    staple += link;
  }
  idxoddbit += muvolume;
  msun U = GAUGE_LOAD<UseTex, atype, Real>(arrayin, idxoddbit, offset);

  cuRNGState localState = state[id];

  msun link = U;
  for (int iter = 0; iter < nhit; iter++) {
    heatBathSUN<Real>(U, staple.dagger(), localState);
    link += U;
  }
  link /= (Real)(nhit + 1);

  state[id] = localState;
  GAUGE_SAVE<atypehit, Real>(arrayout, link, idxoddbit, offset);
}

template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real,
          bool even>
class MultiHit : Tunable {
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
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    if (even) {
      kernel_evenodd<UseTex, atypein, atypeout, Real>
          <<<tp.grid, tp.block, 0, stream>>>(arrayin.GetPtr(),
                                             arrayout.GetPtr(),
                                             randstates.State(), mu, 0, nhit);
      kernel_evenodd<UseTex, atypein, atypeout, Real>
          <<<tp.grid, tp.block, 0, stream>>>(arrayin.GetPtr(),
                                             arrayout.GetPtr(),
                                             randstates.State(), mu, 1, nhit);
    } else
      kernel_multihit_1D_halfrng<UseTex, atypein, atypeout, Real>
          <<<tp.grid, tp.block, 0, stream>>>(arrayin.GetPtr(),
                                             arrayout.GetPtr(),
                                             randstates.State(), mu, nhit);
  }

public:
  MultiHit(gauge &arrayin, gauge &arrayout, RNG &randstates, int mu, int nhit)
      : arrayin(arrayin), arrayout(arrayout), randstates(randstates), mu(mu),
        nhit(nhit) {
    size = 1;
    // Number of threads is equal to the number of space points!
    for (int i = 0; i < NDIMS; i++) {
      size *= PARAMS::Grid[i];
    }
    size = size / 2;
    timesec = 0.0;
  }
  ~MultiHit() {};

  double time() { return timesec; }
  void stat() {
    COUT << "MultiHit:  " << time() << " s\t" << bandwidth() << " GB/s\t"
         << flops() << " GFlops" << endl;
  }
  long long flop() const { return 0; }
  long long bytes() const { return 0; }
  double flops() { return ((double)flop() * 1.0e-9) / timesec; }
  double bandwidth() { return (double)bytes() / (timesec * (double)(1 << 30)); }
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    for (int i = 0; i < NDIMS - 1; i++)
      vol << PARAMS::Grid[i] << "x";
    vol << PARAMS::Grid[NDIMS - 1];
    aux << "threads=" << size << ",prec=" << sizeof(Real);
    string typear = arrayin.ToStringArrayType() + arrayout.ToStringArrayType();
    return TuneKey(vol.str().c_str(), typeid(*this).name(), typear.c_str(),
                   aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << ","
       << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {
    arrayin.Backup();
    randstates.Backup();
  }
  void postTune() {
    arrayin.Restore();
    randstates.Restore();
  }

  void Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
    mtime.start();
#endif
    apply(stream);
#ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
  }
  void Run() { return Run(0); }
};

template <class Real, bool tex, ArrayType atypein, ArrayType atypeout>
void ApplyMultiHit(gauge array, gauge arrayout, RNG &randstates, int nhit) {

  COUT << "MultiHit: nhit = " << nhit << endl;
  Timer mtime;
  mtime.start();
  if (array.EvenOdd() && arrayout.EvenOdd()) {
    MultiHit<tex, atypein, atypeout, Real, true> mhit(
        array, arrayout, randstates, NDIMS - 1, nhit);
    mhit.Run();
    mhit.stat();
  } else if (!array.EvenOdd() && !arrayout.EvenOdd()) {
    MultiHit<tex, atypein, atypeout, Real, false> mhit(
        array, arrayout, randstates, NDIMS - 1, nhit);
    mhit.Run();
    mhit.stat();
  } else
    errorCULQCD("Not defined...\n");
  CUDA_SAFE_DEVICE_SYNC();
  mtime.stop();
  COUT << "Time MultiHit:  " << mtime.getElapsedTimeInSec() << " s" << endl;
}

template <class Real, bool tex>
void ApplyMultiHit(gauge array, gauge arrayout, RNG &randstates, int nhit) {
  if (arrayout.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays in arrayout...\n");
  if (array.Type() == SOA)
    ApplyMultiHit<Real, tex, SOA, SOA>(array, arrayout, randstates, nhit);
  else if (array.Type() == SOA12)
    ApplyMultiHit<Real, tex, SOA12, SOA>(array, arrayout, randstates, nhit);
  else
    errorCULQCD("Not defined for SOA8 arrays in array...\n");
}

template <class Real>
void ApplyMultiHit(gauge array, gauge arrayout, RNG &randstates, int nhit) {

  if (PARAMS::UseTex) {
    GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyMultiHit<Real, true>(array, arrayout, randstates, nhit);
  } else
    ApplyMultiHit<Real, false>(array, arrayout, randstates, nhit);
}
template void ApplyMultiHit<float>(gauges array, gauges arrayout,
                                   RNG &randstates, int nhit);
template void ApplyMultiHit<double>(gauged array, gauged arrayout,
                                    RNG &randstates, int nhit);

} // namespace CULQCD
