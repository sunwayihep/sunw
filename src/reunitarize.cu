
#include <constants.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <index.h>
#include <matrixsun.h>
#include <reunitarize.h>
#include <reunitlink.h>
#include <texture_host.h>
#include <timer.h>
// #include <comm_mpi.h>

using namespace std;

namespace CULQCD {

/**
        @brief Kernel to reunitarize all links.
        @param array array to be reunitarized.
*/
template <bool UseTex, ArrayType atype, class Real>
__global__ void kernel_calc_reunitarize(complex *array, int size) {
  uint id = INDEX1D();
  if (id < size) {
    msun U = GAUGE_LOAD<UseTex, atype, Real>(array, id, size);
    reunit_link<Real>(&U);
    GAUGE_SAVE<atype, Real>(array, U, id, size);
  }
}

template <class Real>
Reunitarize<Real>::Reunitarize(gauge &array) : array(array) {
  SetFunctionPtr();
  size = array.Size();
  timesec = 0.0;
}
template <class Real> void Reunitarize<Real>::SetFunctionPtr() {
  kernel_pointer = NULL;
  tex = PARAMS::UseTex;
  // if(array.EvenOdd()){ //independent of the normal or even/odd indexing
  if (tex) {
#if (NCOLORS == 3)
    if (array.Type() == SOA)
      kernel_pointer = &kernel_calc_reunitarize<true, SOA, Real>;
    if (array.Type() == SOA12)
      kernel_pointer = &kernel_calc_reunitarize<true, SOA12, Real>;
    if (array.Type() == SOA8)
      kernel_pointer = &kernel_calc_reunitarize<true, SOA8, Real>;
#else
    kernel_pointer = &kernel_calc_reunitarize<true, SOA, Real>;
#endif
  } else {
#if (NCOLORS == 3)
    if (array.Type() == SOA)
      kernel_pointer = &kernel_calc_reunitarize<false, SOA, Real>;
    if (array.Type() == SOA12)
      kernel_pointer = &kernel_calc_reunitarize<false, SOA12, Real>;
    if (array.Type() == SOA8)
      kernel_pointer = &kernel_calc_reunitarize<false, SOA8, Real>;
#else
    kernel_pointer = &kernel_calc_reunitarize<false, SOA, Real>;
#endif
  }
  //}
  if (kernel_pointer == NULL)
    errorCULQCD("No kernel Reunitarize function exist for this gauge array...");
}

template <class Real>
void Reunitarize<Real>::apply(const cudaStream_t &stream) {
  TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  kernel_pointer<<<tp.grid, tp.block, 0, stream>>>(array.GetPtr(), size);
}
template <class Real> void Reunitarize<Real>::Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
  mtime.start();
#endif
  // just ensure that the texture was not unbind somewhere...
  if (tex != PARAMS::UseTex) {
    SetFunctionPtr();
  }
  GAUGE_TEXTURE(array.GetPtr(), true);
  apply(stream);
#ifdef TIMMINGS
  CUDA_SAFE_DEVICE_SYNC();
  mtime.stop();
  timesec = mtime.getElapsedTimeInSec();
#endif
}
template <class Real> void Reunitarize<Real>::Run() { Run(0); }
template <class Real> double Reunitarize<Real>::time() { return timesec; }

template <class Real> void Reunitarize<Real>::stat() {
  COUT << "Reunitarize:  " << time() << " s\t" << bandwidth() << " GB/s\t"
       << flops() << " GFlops" << endl;
}

template <class Real> long long Reunitarize<Real>::flop() const {
  // NEEEDDDDD TO RECOUNT THIS PART!!!!!!!!!!!!!!!!!!!!!!!!!
#if (NCOLORS == 3)
  long long ThreadFlop =
      (array.getNumFlop(true) + array.getNumFlop(false) + 126LL);
#else
  unsigned int tmp_gs = 0;
  unsigned int tmp_det = 0;
  for (int i = 0; i < NCOLORS; i++) {
    tmp_gs += i + 1;
    tmp_det += i;
  }
  tmp_det = tmp_gs * NCOLORS * 8 + tmp_det * (NCOLORS * 8 + 11);
  tmp_gs = tmp_gs * NCOLORS * 16 + NCOLORS * (NCOLORS * 6 + 2);
  long long ThreadFlop = (long long)(tmp_gs + tmp_det);
#endif
  return ThreadFlop * size * numnodes();
}
template <class Real> long long Reunitarize<Real>::bytes() const {
  return 2LL * array.getNumParams() * sizeof(Real) * size * numnodes();
}

template <class Real> double Reunitarize<Real>::flops() {
  return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> double Reunitarize<Real>::bandwidth() {
  return (double)bytes() / (timesec * (double)(1 << 30));
}

template class Reunitarize<float>;
template class Reunitarize<double>;

} // namespace CULQCD
