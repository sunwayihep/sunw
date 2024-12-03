

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <comm_mpi.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <index.h>
#include <reduction.h>
#include <texture_host.h>
#include <timer.h>

#include <tune.h>

#include <projectlink.h>
#include <reunitlink.h>

#include <smear/smear.h>

using namespace std;

namespace CULQCD {

// https://arxiv.org/abs/hep-lat/0103029
// Flavor Symmetry and the Static Potential with Hypercubic Blocking
// A. Hasenfratz, F. Knechtli (U. of Colorado)

// NEEDS MORE OPTIMIZATIONS HERE
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// NEEDS MORE OPTIMIZATIONS HERE
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, ArrayType atype, class Real>
__device__ msun inline CalcStaple(complex *array, int x[NDIMS], int mu,
                                  int nu) {
  int muvolume = mu * DEVPARAMS::Volume;
  int dx[NDIMS] = {0};
  int nuvolume = nu * DEVPARAMS::Volume;
  msun link;
  // UP
  link = GAUGE_LOAD<UseTex, atype, Real>(array, Index_ND_NM(x) + nuvolume,
                                         DEVPARAMS::size);
  dx[nu]++;
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_Neig_NM(x, dx) + muvolume, DEVPARAMS::size);
  dx[nu]--;
  dx[mu]++;
  link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
      array, Index_ND_Neig_NM(x, dx) + nuvolume, DEVPARAMS::size);
  msun staple = link;
  dx[mu]--;
  // DOWN
  dx[nu]--;
  link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
      array, Index_ND_Neig_NM(x, dx) + nuvolume, DEVPARAMS::size);
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_Neig_NM(x, dx) + muvolume, DEVPARAMS::size);
  dx[mu]++;
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_Neig_NM(x, dx) + nuvolume, DEVPARAMS::size);
  staple += link;
  return staple;
}

template <bool UseTex, ArrayType atype, class Real>
__device__ msun HYPSmearVbar(complex *array, int idx[NDIMS], int mu, int nu,
                             int rho, int nhits, Real tol) {
  int eta = 0;
  for (int i = 0; i < NDIMS; ++i)
    if (i != mu && i != nu && i != rho)
      eta = i;
  msun vbar = CalcStaple<UseTex, atype, Real>(array, idx, mu, eta);
  vbar *= (0.5 * DEVPARAMS::hypalpha3);
  msun link = GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_NM(idx) + mu * DEVPARAMS::Volume, DEVPARAMS::size);
  vbar += link * (1.0 - DEVPARAMS::hypalpha3);
#if defined(PROJECT_LINK_START_WREUNIT)
  link = vbar;
  reunit_link<Real>(&link);
#endif
  project_link<Real>(link, vbar, nhits, tol);
  return link;
}

template <bool UseTex, ArrayType atype, class Real>
__device__ msun HYPSmearVtil(complex *array, int idx[NDIMS], int mu, int nu,
                             int nhits, Real tol) {
  msun link;
  msun vtil = msun::zero();
  for (int rho = 0; rho < NDIMS; ++rho)
    if (rho != mu && rho != nu) {
      link = HYPSmearVbar<UseTex, atype, Real>(array, idx, rho, nu, mu, nhits,
                                               tol);
      int y[NDIMS];
      Index_ND_Neig_NM(y, idx, rho, 1);
      link *=
          HYPSmearVbar<UseTex, atype, Real>(array, y, mu, rho, nu, nhits, tol);
      Index_ND_Neig_NM(y, idx, mu, 1);
      link *=
          HYPSmearVbar<UseTex, atype, Real>(array, y, rho, nu, mu, nhits, tol)
              .dagger();
      vtil += link;
      Index_ND_Neig_NM(y, idx, rho, -1);
      link =
          HYPSmearVbar<UseTex, atype, Real>(array, y, rho, nu, mu, nhits, tol)
              .dagger();
      link *=
          HYPSmearVbar<UseTex, atype, Real>(array, y, mu, rho, nu, nhits, tol);
      Index_ND_Neig_NM(y, idx, rho, -1, mu, 1);
      link *=
          HYPSmearVbar<UseTex, atype, Real>(array, y, rho, nu, mu, nhits, tol);
      vtil += link;
    }
  link = GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_NM(idx) + mu * DEVPARAMS::Volume, DEVPARAMS::size);
  vtil *= (0.25 * DEVPARAMS::hypalpha2);
  vtil += link * (1.0 - DEVPARAMS::hypalpha2);
#if defined(PROJECT_LINK_START_WREUNIT)
  link = vtil;
  reunit_link<Real>(&link);
#endif
  project_link<Real>(link, vtil, nhits, tol);
  return link;
}

template <bool UseTex, ArrayType atype, class Real>
__device__ msun HYPSmearV(complex *array, int idx[NDIMS], int mu, int nhits,
                          Real tol) {

  msun v = msun::zero();
  msun link;
  for (int nu = 0; nu < NDIMS; ++nu) {
    if (mu != nu) {
      link = HYPSmearVtil<UseTex, atype, Real>(array, idx, nu, mu, nhits, tol);
      int y[NDIMS];
      Index_ND_Neig_NM(y, idx, nu, 1);
      link *= HYPSmearVtil<UseTex, atype, Real>(array, y, mu, nu, nhits, tol);
      Index_ND_Neig_NM(y, idx, mu, 1);
      link *= HYPSmearVtil<UseTex, atype, Real>(array, y, nu, mu, nhits, tol)
                  .dagger();
      v += link;
      Index_ND_Neig_NM(y, idx, nu, -1);
      link = HYPSmearVtil<UseTex, atype, Real>(array, y, nu, mu, nhits, tol)
                 .dagger();
      link *= HYPSmearVtil<UseTex, atype, Real>(array, y, mu, nu, nhits, tol);
      Index_ND_Neig_NM(y, idx, nu, -1, mu, 1);
      link *= HYPSmearVtil<UseTex, atype, Real>(array, y, nu, mu, nhits, tol);
      v += link;
    }
  }
  link = GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_NM(idx) + mu * DEVPARAMS::Volume, DEVPARAMS::size);
  v *= DEVPARAMS::hypalpha1 / 6.0;
  v += link * (1.0 - DEVPARAMS::hypalpha1);
#if defined(PROJECT_LINK_START_WREUNIT)
  link = v;
  reunit_link<Real>(&link);
#endif
  project_link<Real>(link, v, nhits, tol);
  return link;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Real> struct HYPArg {
  complex *arrayin;
  complex *arrayout;
  Real tol;
  int nhits;
};

template <bool UseTex, ArrayType atype, class Real>
__global__ void kernel_HYPSmear(HYPArg<Real> arg, int mu) {
  int id = INDEX1D();
  if (id >= DEVPARAMS::Volume)
    return;
  int x[NDIMS];
  Index_ND_NM(id, x);
  msun vhyp =
      HYPSmearV<UseTex, atype, Real>(arg.arrayin, x, mu, arg.nhits, arg.tol);
  GAUGE_SAVE<atype, Real>(arg.arrayout, vhyp, id + mu * DEVPARAMS::Volume,
                          DEVPARAMS::size);
}

template <bool UseTex, ArrayType atypein, class Real> class ApplyHYP : Tunable {
private:
  gauge arrayin;
  gauge arrayout;
  HYPArg<Real> arg;
  int size;
  double timesec;
  int mu;
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
    kernel_HYPSmear<UseTex, atypein, Real>
        <<<tp.grid, tp.block, 0, stream>>>(arg, mu);
  }

public:
  ApplyHYP(gauge &arrayin, gauge &arrayout, int nhits, Real tol)
      : arrayin(arrayin), arrayout(arrayout) {
    size = 1;
    // Number of threads is equal to the number of space points!
    for (int i = 0; i < NDIMS; i++) {
      size *= PARAMS::Grid[i];
    }
    size = size;
    timesec = 0.0;
    arg.arrayin = arrayin.GetPtr();
    arg.arrayout = arrayout.GetPtr();
    arg.nhits = nhits;
    arg.tol = tol;
    mu = 0;
  }
  ~ApplyHYP() {};

  void SetDir(int muin) { mu = muin; };

  double time() { return timesec; }
  void stat() {
    COUT << "ApplyHYP:  " << time() << " s\t" << bandwidth() << " GB/s\t"
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
  void preTune() {}
  void postTune() {}

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

template <class Real>
void ApplyHYPinSpace(gauge array, int steps, int nhits, Real tol,
                     ParamHYP hyp) {
  cout << "Apply HYP Smearing in Space with steps = " << steps
       << " nhit = " << nhits << " tol = " << tol << endl;
  if (array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if (array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");

  Timer a0;
  a0.start();

  hyp.print();
  hyp.copyToGPUMem();

  gauge arrayout(array.Type(), Device, array.Size(), array.EvenOdd());
  arrayout.Copy(array);
  const ArrayType atypein = SOA;
  if (PARAMS::UseTex) {
    GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyHYP<true, atypein, Real> hyp(array, arrayout, nhits, tol);
    for (int st = 0; st < steps; st++) {
      for (int mu = 0; mu < NDIMS - 1; mu++) {
        hyp.SetDir(mu);
        hyp.Run();
      }
      array.Copy(arrayout);
    }
    hyp.stat();
  } else {
    ApplyHYP<false, atypein, Real> hyp(array, arrayout, nhits, tol);
    for (int st = 0; st < steps; st++) {
      for (int mu = 0; mu < NDIMS - 1; mu++) {
        hyp.SetDir(mu);
        hyp.Run();
      }
      array.Copy(arrayout);
    }
    hyp.stat();
  }
  arrayout.Release();
  a0.stop();
  COUT << "Time to apply HYP smearing in space: " << a0.getElapsedTime() << " s"
       << endl;
}
template void ApplyHYPinSpace<float>(gauges array, int steps, int nhits,
                                     float tol, ParamHYP hyp);
template void ApplyHYPinSpace<double>(gauged array, int steps, int nhits,
                                      double tol, ParamHYP hyp);

template <class Real>
void ApplyHYPinTime(gauge array, int steps, int nhits, Real tol, ParamHYP hyp) {
  cout << "Apply HYP Smearing in Time with steps = " << steps
       << " nhit = " << nhits << " tol = " << tol << endl;
  if (array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if (array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");

  Timer a0;
  a0.start();

  hyp.print();
  hyp.copyToGPUMem();

  gauge arrayout(array.Type(), Device, array.Size(), array.EvenOdd());
  arrayout.Copy(array);
  const ArrayType atypein = SOA;
  if (PARAMS::UseTex) {
    GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyHYP<true, atypein, Real> hyp(array, arrayout, nhits, tol);
    hyp.SetDir(NDIMS - 1);
    for (int st = 0; st < steps; st++) {
      hyp.Run();
      array.Copy(arrayout);
    }
    hyp.stat();
  } else {
    ApplyHYP<false, atypein, Real> hyp(array, arrayout, nhits, tol);
    hyp.SetDir(NDIMS - 1);
    for (int st = 0; st < steps; st++) {
      hyp.Run();
      array.Copy(arrayout);
    }
    hyp.stat();
  }
  arrayout.Release();
  a0.stop();
  COUT << "Time to apply HYP smearing in time: " << a0.getElapsedTime() << " s"
       << endl;
}
template void ApplyHYPinTime<float>(gauges array, int steps, int nhits,
                                    float tol, ParamHYP hyp);
template void ApplyHYPinTime<double>(gauged array, int steps, int nhits,
                                     double tol, ParamHYP hyp);

ParamHYP::ParamHYP() {
  alpha1 = 0.75;
  alpha2 = 0.6;
  alpha3 = 0.3;
}
ParamHYP::ParamHYP(float alpha1, float alpha2, float alpha3)
    : alpha1(alpha1), alpha2(alpha2), alpha3(alpha3) {}
ParamHYP::~ParamHYP() {};

void ParamHYP::setDefault() {
  alpha1 = 0.75;
  alpha2 = 0.6;
  alpha3 = 0.3;
  /*alpha1 = 1.0;
  alpha2 = 0.5;
  alpha3 = 0.5;*/
}
void ParamHYP::set(float _alpha1, float _alpha2, float _alpha3) {
  alpha1 = _alpha1;
  alpha2 = _alpha2;
  alpha3 = _alpha3;
}
void ParamHYP::print() {
  COUT << "HYP params -> alpha1: " << alpha1 << "\talpha2: " << alpha2
       << "\talpha3: " << alpha3 << endl;
}
void ParamHYP::copyToGPUMem() { copyHYPSmearConstants(alpha1, alpha2, alpha3); }

} // namespace CULQCD
