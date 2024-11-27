

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <comm_mpi.h>
#include <constants.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <index.h>
#include <meas/plaquette.h>
#include <reduction.h>
#include <texture_host.h>
#include <timer.h>

#include <cudaAtomic.h>

#include <cub/cub.cuh>

using namespace std;

namespace CULQCD {

inline __host__ __device__ int neighborNormalIndex(int id, int mu, int lmu) {
  int x[NDIMS];
  Index_ND_NM(id, x);
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] += param_border(i);
#endif
  x[mu] = (x[mu] + lmu) % param_GridG(mu);

  return Index_ND_NM(x);
}

// kernel to calculate the plaquette at each site of the lattice in EvenOdd
// order
template <int blockSize, bool UseTex, ArrayType atype, class Real>
__global__ void kernel_calc_plaquette_normal_cub(PlaqArg<Real> arg) {
  uint id = INDEX1D();

  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  complex plaq = complex::zero();

  if (id < param_Volume()) {
#ifdef MULTI_GPU
    int x[NDIMS];
    Index_ND_NM(id, x);
    for (int i = 0; i < NDIMS; i++)
      x[i] += param_border(i);
    int mustride = DEVPARAMS::VolumeG;
    int offset = mustride * NDIMS;
#else
    int mustride = DEVPARAMS::Volume;
    int offset = mustride * NDIMS;

#endif
    //------------------------------------------------------------------------
    // Calculate space-time plaquettes, stored in the real real of plaquette
    // array
    //------------------------------------------------------------------------
    msun link, link1;
    // #pragma unroll
    for (int mu = 0; mu < NDIMS; mu++) {
      link1 = GAUGE_LOAD<UseTex, atype, Real>(arg.pgauge, id + mu * mustride,
                                              offset);
      int newidmu1 = neighborNormalIndex(id, mu, 1);
      // #pragma unroll
      for (int nu = (mu + 1); nu < NDIMS; nu++) {
        link = GAUGE_LOAD<UseTex, atype, Real>(
            arg.pgauge, newidmu1 + nu * mustride, offset);
        link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.pgauge, neighborNormalIndex(id, nu, 1) + mu * mustride, offset);
        link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.pgauge, id + nu * mustride, offset);
        if (nu == (NDIMS - 1))
          plaq.imag() += (link1 * link).realtrace();
        else
          plaq.real() += (link1 * link).realtrace();
      }
    }
// average plaqs over different spatial and time directions
#if (NDIMS > 2)
    plaq.imag() /= TOTAL_NUM_TPLAQS;
    plaq.real() /= TOTAL_NUM_SPLAQS;
#endif
  }
  complex aggregate = BlockReduce(temp_storage).Reduce(plaq, Summ<complex>());
  if (threadIdx.x == 0)
    CudaAtomicAdd(arg.plaq, aggregate);
}

// kernel to calculate the plaquette at each site of the lattice in EvenOdd
// order
template <int blockSize, bool UseTex, ArrayType atype, class Real>
__global__ void kernel_calc_plaquette_evenodd_cub(PlaqArg<Real> arg) {
  uint idd = INDEX1D();

  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  complex plaq = complex::zero();

  if (idd < param_Volume()) {

    int oddbit = 0;
    int id = idd;
    if (idd >= param_HalfVolume()) {
      oddbit = 1;
      id = idd - param_HalfVolume();
    }
#ifdef MULTI_GPU
    int x[NDIMS];
    Index_ND_EO(x, id, oddbit);
    for (int i = 0; i < NDIMS; i++)
      x[i] += param_border(i);
    int idxoddbit = Index_ND_NM(x, DEVPARAMS::GridWGhost) >> 1;
    idxoddbit += oddbit * param_HalfVolumeG();
    int mustride = DEVPARAMS::VolumeG;
    int offset = mustride * NDIMS;
#else
    int mustride = DEVPARAMS::Volume;
    int offset = mustride * NDIMS;
    int idxoddbit = id + oddbit * param_HalfVolume();
    // int idxoddbit = idd; //cuda reports error: misaligned address LOL

#endif
    //------------------------------------------------------------------------
    // Calculate space-time plaquettes, stored in the real real of plaquette
    // array
    //------------------------------------------------------------------------
    msun link, link1;
    // #pragma unroll
    for (int mu = 0; mu < NDIMS; mu++) {
      link1 = GAUGE_LOAD<UseTex, atype, Real>(
          arg.pgauge, idxoddbit + mu * mustride, offset);
      int newidmu1 = Index_ND_Neig_EO(id, oddbit, mu, 1);
      // #pragma unroll
      for (int nu = (mu + 1); nu < NDIMS; nu++) {
        link = GAUGE_LOAD<UseTex, atype, Real>(
            arg.pgauge, newidmu1 + nu * mustride, offset);
        link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.pgauge, Index_ND_Neig_EO(id, oddbit, nu, 1) + mu * mustride,
            offset);
        link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.pgauge, idxoddbit + nu * mustride, offset);
        if (nu == (NDIMS - 1))
          plaq.imag() += (link1 * link).realtrace();
        else
          plaq.real() += (link1 * link).realtrace();
      }
    }
// average plaqs over different spatial and time directions
#if (NDIMS > 2)
    plaq.imag() /= TOTAL_NUM_TPLAQS;
    plaq.real() /= TOTAL_NUM_SPLAQS;
#endif
  }
  complex aggregate = BlockReduce(temp_storage).Reduce(plaq, Summ<complex>());
  if (threadIdx.x == 0)
    CudaAtomicAdd(arg.plaq, aggregate);
}

template <class Real>
PlaquetteCUB<Real>::PlaquetteCUB(gauge &array) : array(array) {

  functionName = "Plaquette";
  plaq_value = complex::zero();
  size = 1;
  for (int i = 0; i < NDIMS; i++) {
    grid[i] = PARAMS::Grid[i];
    size *= PARAMS::Grid[i];
  }
  timesec = 0.0;
  arg.pgauge = array.GetPtr();
  arg.plaq = (complex *)dev_malloc(sizeof(complex));
}

template <class Real>
void PlaquetteCUB<Real>::apply(const cudaStream_t &stream) {
  TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  CUDA_SAFE_CALL(cudaMemset(arg.plaq, 0, sizeof(complex)));
  if (array.EvenOdd()) {
    if (PARAMS::UseTex) {
      // just ensure that the texture was not unbind somewhere...
      BIND_GAUGE_TEXTURE(array.GetPtr());
#if (NCOLORS == 3)
      if (array.Type() == SOA)
        LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, true,
                      SOA, Real);
      if (array.Type() == SOA12)
        LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, true,
                      SOA12, Real);
      if (array.Type() == SOA8)
        LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, true,
                      SOA8, Real);
#else
      LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, true,
                    SOA, Real);
#endif
    } else {
#if (NCOLORS == 3)
      if (array.Type() == SOA)
        LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, false,
                      SOA, Real);
      if (array.Type() == SOA12)
        LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, false,
                      SOA12, Real);
      if (array.Type() == SOA8)
        LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, false,
                      SOA8, Real);
#else
      LAUNCH_KERNEL(kernel_calc_plaquette_evenodd_cub, tp, stream, arg, false,
                    SOA, Real);
#endif
    }
  } else {
    if (PARAMS::UseTex) {
      // just ensure that the texture was not unbind somewhere...
      BIND_GAUGE_TEXTURE(array.GetPtr());
#if (NCOLORS == 3)
      if (array.Type() == SOA)
        LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, true,
                      SOA, Real);
      if (array.Type() == SOA12)
        LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, true,
                      SOA12, Real);
      if (array.Type() == SOA8)
        LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, true,
                      SOA8, Real);
#else
      LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, true,
                    SOA, Real);
#endif
    } else {
#if (NCOLORS == 3)
      if (array.Type() == SOA)
        LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, false,
                      SOA, Real);
      if (array.Type() == SOA12)
        LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, false,
                      SOA12, Real);
      if (array.Type() == SOA8)
        LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, false,
                      SOA8, Real);
#else
      LAUNCH_KERNEL(kernel_calc_plaquette_normal_cub, tp, stream, arg, false,
                    SOA, Real);
#endif
    }
  }
}

template <class Real>
complex PlaquetteCUB<Real>::Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
  plaqtime.start();
#endif
  apply(stream);
  CUDA_SAFE_DEVICE_SYNC(); //
  CUT_CHECK_ERROR("Kernel execution failed");
  CUDA_SAFE_CALL(cudaMemcpy(&plaq_value, arg.plaq, sizeof(complex),
                            cudaMemcpyDeviceToHost));
  plaq_value /= (Real)(NCOLORS * size);
#ifdef MULTI_GPU
  comm_Allreduce(&plaq_value);
  plaq_value /= numnodes();
#endif
#ifdef TIMMINGS
  CUDA_SAFE_DEVICE_SYNC();
  plaqtime.stop();
  timesec = plaqtime.getElapsedTimeInSec();
#endif
  return plaq_value;
}
template <class Real> complex PlaquetteCUB<Real>::Run() { return Run(0); }

template <class Real> double PlaquetteCUB<Real>::time() { return timesec; }

template <class Real> void PlaquetteCUB<Real>::stat() {
  COUT << "Plaquette:  " << time() << " s\t" << bandwidth() << " GB/s\t"
       << flops() << " GFlops" << endl;
}

template <class Real> void PlaquetteCUB<Real>::printValue() {
  Real resf = (plaq_value.real() + plaq_value.imag()) / 2.0;
  //.real() space  and .imag() time part
  printfCULQCD("Plaquette: < %.12e : %.12e > :: mean: %.12e\n",
               plaq_value.real(), plaq_value.imag(), resf);
}

template <class Real> long long PlaquetteCUB<Real>::flop() const {
#ifdef MULTI_GPU
  return NCOLORS * NCOLORS * NCOLORS * 120LL * size * numnodes();
#else
  return NCOLORS * NCOLORS * NCOLORS * 120LL * size * numnodes();
#endif
}
template <class Real> long long PlaquetteCUB<Real>::bytes() const {
#ifdef MULTI_GPU
  return (22LL * array.getNumParams() + 4LL) * size * numnodes() * sizeof(Real);
#else
  return (22LL * array.getNumParams() + 4LL) * size * sizeof(Real);
#endif
}

template <class Real> double PlaquetteCUB<Real>::flops() {
  return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> double PlaquetteCUB<Real>::bandwidth() {
  return (double)bytes() / (timesec * (double)(1 << 30));
}

template class PlaquetteCUB<float>;
template class PlaquetteCUB<double>;

} // namespace CULQCD
