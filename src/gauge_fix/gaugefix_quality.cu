
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <typeinfo>

#include <comm_mpi.h>
#include <complex.h>
#include <cuda_common.h>
#include <gaugefix/gaugefix.h>
#include <matrixsun.h>

#include <constants.h>
#include <device_load_save.h>
#include <index.h>
#include <reduction.h>
#include <texture.h>
#include <texture_host.h>
#include <timer.h>
#include <tune.h>

namespace CULQCD {

#ifdef USE_GAUGE_FIX

template <int DIR, bool UseTex, ArrayType atype, class Real>
__global__ void kernel_calc_Fg_theta_dir(complex *array, complex *res_save) {
  int idd = INDEX1D();
  if (idd < DEVPARAMS::Volume) {
    int id = idd;
    int oddbit = 0;
    if (idd >= DEVPARAMS::HalfVolume) {
      id -= DEVPARAMS::HalfVolume;
      oddbit = 1;
    }
    int offset = DEVPARAMS::VolumeG * 4;
    int idx = EOIndeX(id, oddbit);
    msun delta = msun::zero();
    // Uplinks
    for (int nu = 0; nu < DIR; nu++)
      delta -= GAUGE_LOAD<UseTex, atype, Real>(
          array, idx + nu * DEVPARAMS::VolumeG, offset);
    complex res;
    // Fg (sum_DIR uplinks)
    res.real() = -delta.realtrace();
    // Downlinks
    for (int nu = 0; nu < DIR; nu++)
      delta += GAUGE_LOAD<UseTex, atype, Real>(
          array,
          neighborEOIndexMinusOne(id, oddbit, nu) + nu * DEVPARAMS::VolumeG,
          offset);
    delta = (delta - delta.dagger()).subtraceunit();
    // theta
    res.imag() = realtraceUVdagger(delta, delta);
    res_save[idd] = res;
  }
}

template <int DIR, bool UseTex, ArrayType atype, class Real>
void GaugeFixQuality<DIR, UseTex, atype, Real>::apply(
    const cudaStream_t &stream) {
  TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  kernel_calc_Fg_theta_dir<DIR, UseTex, atype, Real>
      <<<tp.grid, tp.block, 0, stream>>>(array.GetPtr(), sum);
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
GaugeFixQuality<DIR, UseTex, atype, Real>::GaugeFixQuality(gauge &array)
    : array(array) {
  if (array.Type() != atype)
    errorCULQCD("gauge array type and template types do not match...");
  if (!array.EvenOdd())
    errorCULQCD("gauge array must be a even/odd array...");
  functionName = "GaugeFixQuality";
  value = complex::zero();
  size = 1;
  for (int i = 0; i < 4; i++) {
    grid[i] = PARAMS::Grid[i];
    size *= PARAMS::Grid[i];
  }
  sum = (complex *)dev_malloc(size * sizeof(complex));
  timesec = 0.0;
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
GaugeFixQuality<DIR, UseTex, atype, Real>::~GaugeFixQuality() {
  dev_free(sum);
};
template <int DIR, bool UseTex, ArrayType atype, class Real>
complex
GaugeFixQuality<DIR, UseTex, atype, Real>::Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
  mtime.start();
#endif
  if (UseTex)
    BIND_GAUGE_TEXTURE(array.GetPtr());
  apply(stream);
  value = reduction<complex>(sum, size, stream);
  value /= (Real)(PARAMS::Volume * NCOLORS);
  value.real() /= (Real)DIR;
#ifdef MULTI_GPU
  comm_Allreduce(&value);
  value /= numnodes();
#endif
#ifdef TIMMINGS
  CUDA_SAFE_DEVICE_SYNC();
  mtime.stop();
  timesec = mtime.getElapsedTimeInSec();
#endif
  return value;
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
complex GaugeFixQuality<DIR, UseTex, atype, Real>::Run() {
  return Run(0);
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
double GaugeFixQuality<DIR, UseTex, atype, Real>::flops() {
  return ((double)flop() * 1.0e-9) / timesec;
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
double GaugeFixQuality<DIR, UseTex, atype, Real>::bandwidth() {
  return (double)bytes() / (timesec * (double)(1 << 30));
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
long long GaugeFixQuality<DIR, UseTex, atype, Real>::flop() const {
  long long arrayflops = 2LL * DIR * array.getNumFlop(true);
  return (arrayflops + 2LL * NCOLORS * NCOLORS * (DIR + 1) +
          4LL * NCOLORS * (1 + NCOLORS)) *
         size * numnodes();
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
long long GaugeFixQuality<DIR, UseTex, atype, Real>::bytes() const {
  return (2LL * DIR * array.getNumParams() + 2LL) * size * sizeof(Real) *
         numnodes();
}

template <int DIR, bool UseTex, ArrayType atype, class Real>
double GaugeFixQuality<DIR, UseTex, atype, Real>::time() {
  return timesec;
}

template <int DIR, bool UseTex, ArrayType atype, class Real>
void GaugeFixQuality<DIR, UseTex, atype, Real>::stat() {
  COUT << "GaugeFixQuality:  " << time() << " s\t" << bandwidth() << " GB/s\t"
       << flops() << " GFlops" << endl;
}
template <int DIR, bool UseTex, ArrayType atype, class Real>
void GaugeFixQuality<DIR, UseTex, atype, Real>::printValue() {
  printfCULQCD("GaugeFixQuality:Fg = %.12e\ttheta = %.12e\n", value.real(),
               value.imag());
}

#if (NCOLORS == 3)
template class GaugeFixQuality<4, true, SOA, float>;
template class GaugeFixQuality<4, true, SOA12, float>;
template class GaugeFixQuality<4, true, SOA8, float>;

template class GaugeFixQuality<4, true, SOA, double>;
template class GaugeFixQuality<4, true, SOA12, double>;
template class GaugeFixQuality<4, true, SOA8, double>;

template class GaugeFixQuality<4, false, SOA, float>;
template class GaugeFixQuality<4, false, SOA12, float>;
template class GaugeFixQuality<4, false, SOA8, float>;

template class GaugeFixQuality<4, false, SOA, double>;
template class GaugeFixQuality<4, false, SOA12, double>;
template class GaugeFixQuality<4, false, SOA8, double>;

template class GaugeFixQuality<3, true, SOA, float>;
template class GaugeFixQuality<3, true, SOA12, float>;
template class GaugeFixQuality<3, true, SOA8, float>;

template class GaugeFixQuality<3, true, SOA, double>;
template class GaugeFixQuality<3, true, SOA12, double>;
template class GaugeFixQuality<3, true, SOA8, double>;

template class GaugeFixQuality<3, false, SOA, float>;
template class GaugeFixQuality<3, false, SOA12, float>;
template class GaugeFixQuality<3, false, SOA8, float>;

template class GaugeFixQuality<3, false, SOA, double>;
template class GaugeFixQuality<3, false, SOA12, double>;
template class GaugeFixQuality<3, false, SOA8, double>;
#elif (NCOLORS > 3)
template class GaugeFixQuality<4, true, SOA, float>;
template class GaugeFixQuality<4, true, SOA, double>;

template class GaugeFixQuality<4, false, SOA, float>;
template class GaugeFixQuality<4, false, SOA, double>;

template class GaugeFixQuality<3, true, SOA, float>;
template class GaugeFixQuality<3, true, SOA, double>;

template class GaugeFixQuality<3, false, SOA, float>;
template class GaugeFixQuality<3, false, SOA, double>;
#else
#error Code not done for NCOLORS < 3
#endif

#endif

} // namespace CULQCD
