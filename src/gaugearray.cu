
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <complex.h>
#include <constants.h>
#include <cuda_common.h>
#include <device_PHB_OVR.h>
#include <device_load_save.h>
#include <gaugearray.h>
#include <index.h>
#include <matrixsun.h>
#include <reunitlink.h>

#include <exchange.h>

using namespace std;

namespace CULQCD {

template <class Real>
__global__ void kernel_copybetweenArraytypes(_gauge<Real> arrayin,
                                             bool In_evenodd,
                                             _gauge<Real> arrayout,
                                             bool Out_evenodd, int size) {
  uint idd = INDEX1D();
  if (idd >= size)
    return;
  int mu = idd / DEVPARAMS::Volume;
  int id = idd % DEVPARAMS::Volume;

  int id_in = idd;
  int x[4];
  if (In_evenodd) {
    int parity = 0;
    if (id >= DEVPARAMS::HalfVolume) {
      id -= DEVPARAMS::HalfVolume;
      parity = 1;
    }
    int za = (id / (DEVPARAMS::Grid[0] / 2));
    int zb = (za / DEVPARAMS::Grid[1]);
    x[1] = za - zb * DEVPARAMS::Grid[1];
    x[3] = (zb / DEVPARAMS::Grid[2]);
    x[2] = zb - x[3] * DEVPARAMS::Grid[2];
    int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * id + x1odd) - za * DEVPARAMS::Grid[0];
  } else {
    x[3] = id / (DEVPARAMS::Grid[0] * DEVPARAMS::Grid[1] * DEVPARAMS::Grid[2]);
    x[2] =
        (id / (DEVPARAMS::Grid[0] * DEVPARAMS::Grid[1])) % DEVPARAMS::Grid[2];
    x[1] = (id / DEVPARAMS::Grid[0]) % DEVPARAMS::Grid[1];
    x[0] = id % DEVPARAMS::Grid[0];
  }
  id =
      (x[0] + (x[1] + (x[2] + x[3] * DEVPARAMS::Grid[2]) * DEVPARAMS::Grid[1]) *
                  DEVPARAMS::Grid[0]);
  int id_out = id;
  if (Out_evenodd) {
    id_out = id / 2 + ((x[0] + x[1] + x[2] + x[3]) & 1) * DEVPARAMS::HalfVolume;
  }
  arrayout.Set(arrayin.Get(id_in), id_out + mu * DEVPARAMS::Volume);
}

template <class Real>
void GPU_COPY_EO_NORMAL_TYPES(_gauge<Real> arrayin, _gauge<Real> &arrayout) {
  if (arrayin.Size() == arrayout.Size() && numnodes() == 1) {
    dim3 threads(128, 1, 1);
    dim3 blocks = GetBlockDim(threads.x, arrayin.Size());
    kernel_copybetweenArraytypes<Real>
        <<<blocks, threads>>>(arrayin, arrayin.EvenOdd(), arrayout,
                              arrayout.EvenOdd(), arrayin.Size());
    cudaCheckError("GPUCopyArrayTypes: Kernel execution failed");
  } else
    errorCULQCD("Cannot copy between arrays with different sizes...\n");
}
template void GPU_COPY_EO_NORMAL_TYPES<float>(_gauge<float> arrayin,
                                              _gauge<float> &arrayout);
template void GPU_COPY_EO_NORMAL_TYPES<double>(_gauge<double> arrayin,
                                               _gauge<double> &arrayout);

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

template <class Real>
__global__ void kernel_copybetweentypes(_gauge<Real> arrayin,
                                        _gauge<Real> arrayout, int size) {
  uint idd = INDEX1D();
  if (idd >= size)
    return;
  arrayout.Set(arrayin.Get(idd), idd);
}

/**
   @brief Device copy arrays
   @param arrayin source gauge field
   @param arrayout destination gauge field
*/
template <class Real>
void GPUCopyArrayTypes(_gauge<Real> arrayin, _gauge<Real> &arrayout) {
  if (arrayin.Size() == arrayout.Size()) {
    dim3 threads(128, 1, 1);
    dim3 blocks = GetBlockDim(threads.x, arrayin.Size());
    kernel_copybetweentypes<Real>
        <<<blocks, threads>>>(arrayin, arrayout, arrayin.Size());
    cudaCheckError("GPUCopyArrayTypes: Kernel execution failed");
  } else
    errorCULQCD("Cannot copy between arrays with different sizes...\n");
}
template void GPUCopyArrayTypes<float>(_gauge<float> arrayin,
                                       _gauge<float> &arrayout);
template void GPUCopyArrayTypes<double>(_gauge<double> arrayin,
                                        _gauge<double> &arrayout);

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

template <class Real> void _gauge<Real>::Copy(_gauge<Real> &from_gauge) {
  if ((from_gauge.Size() == size && from_gauge.Size() > 0) || size == 0) {
    if (size == 0)
      Allocate(from_gauge.Size());
    if (from_gauge.EvenOdd() && EvenOdd()) {
      if ((from_gauge.atype == atype)) {
        cudaMemcpyKind cptype = copytype(from_gauge.Mode(), mode);
        switch (atype) {
        case SOA:
          cudaSafeCall(cudaMemcpy(array, from_gauge.array,
                                  NCOLORS * NCOLORS * size * sizeof(complex),
                                  cptype));
          COUT << "Copied ARRAY with size: "
               << (float)(NCOLORS * NCOLORS * size * sizeof(complex)) /
                      (float)(1048576)
               << " MB" << std::endl;
          break;
        case SOA12:
          cudaSafeCall(cudaMemcpy(array, from_gauge.array,
                                  6 * size * sizeof(complex), cptype));
          COUT << "Copied ARRAY with size: "
               << (float)(6 * size * sizeof(complex)) / (float)(1048576)
               << " MB" << std::endl;
          break;
        case SOA12A:
          cudaSafeCall(cudaMemcpy(array, from_gauge.array,
                                  6 * size * sizeof(complex), cptype));
          COUT << "Copied ARRAY with size: "
               << (float)(6 * size * sizeof(complex)) / (float)(1048576)
               << " MB" << std::endl;
          break;
        case SOA8:
          cudaSafeCall(cudaMemcpy(array, from_gauge.array,
                                  4 * size * sizeof(complex), cptype));
          COUT << "Copied ARRAY with size: "
               << (float)(4 * size * sizeof(complex)) / (float)(1048576)
               << " MB" << std::endl;
          break;
        }
      } else {
        // Call kernel function to copy from device arrays
        if (from_gauge.mode == mode && mode == Device) {
          GPUCopyArrayTypes<Real>(from_gauge, *this);
        }
        // copy between host arrays
        else if ((from_gauge.mode == mode && mode == Host)) {
          for (int i = 0; i < size; i++)
            Set(from_gauge.Get(i), i);
        } else {
          errorCULQCD("Not yet implemented copy between SOA's types "
                      "Host<->Device. Aborting...\n");
        }
      }
    } else if ((!from_gauge.EvenOdd() || !EvenOdd()) &&
               (from_gauge.mode == mode && mode == Device)) {
      // COUT << "Copying between normal or odd/even order to odd/even or normal
      // order." << std::endl;
      if ((from_gauge.border || border) && numnodes() > 1)
        errorCULQCD("Cannot copy. Arrays with borders...\nNot implemented "
                    "yet...\nAborting...\n");
      GPU_COPY_EO_NORMAL_TYPES<Real>(from_gauge, *this);
    } else
      errorCULQCD("Cannot copy. Arrays with different sizes... Aborting...\n");
  } else
    errorCULQCD("Cannot copy. Arrays with different sizes... Aborting...\n");
}
template void _gauge<float>::Copy(_gauge<float> &from_gauge);
template void _gauge<double>::Copy(_gauge<double> &from_gauge);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class RealIn, class RealOut>
__global__ void kernel_CopyBetweentypes(_gauge<RealIn> arrayin,
                                        _gauge<RealOut> arrayout, int size) {
  uint idd = INDEX1D();
  if (idd >= size)
    return;
  _matrixsun<RealIn, NCOLORS> tmp = arrayin.Get(idd);
  _matrixsun<RealOut, NCOLORS> out;
  for (int i = 0; i < NCOLORS; i++)
    for (int j = 0; j < NCOLORS; j++) {
      out.e[i][j].real() = (RealOut)tmp.e[i][j].real();
      out.e[i][j].imag() = (RealOut)tmp.e[i][j].imag();
    }
  arrayout.Set(out, idd);
}

void GaugeCopy(_gauge<double> arrayin, _gauge<float> &arrayout) {
  if ((arrayin.Size() == arrayout.Size() && arrayout.Size() > 0) ||
      arrayout.Size() == 0) {
    if (arrayout.Size() == 0)
      arrayout.Allocate(arrayin.Size());
    if (!arrayin.EvenOdd() || !arrayout.EvenOdd())
      errorCULQCD("Not implemented yet. Aborting...\n");
    if ((arrayin.Mode() == arrayout.Mode() && arrayin.Mode() == Device) &&
        (arrayin.EvenOdd() == arrayout.EvenOdd())) {
      if (arrayin.Size() == arrayout.Size()) {
        dim3 threads(128, 1, 1);
        dim3 blocks = GetBlockDim(threads.x, arrayin.Size());
        kernel_CopyBetweentypes<double, float>
            <<<blocks, threads>>>(arrayin, arrayout, arrayin.Size());
        cudaCheckError("kernel_CopyBetweentypes: Kernel execution failed");
      } else
        errorCULQCD("Cannot copy between arrays with different sizes...\n");
    }
  }
}
void GaugeCopy(_gauge<float> arrayin, _gauge<double> &arrayout) {
  if ((arrayin.Size() == arrayout.Size() && arrayout.Size() > 0) ||
      arrayout.Size() == 0) {
    if (arrayout.Size() == 0)
      arrayout.Allocate(arrayin.Size());
    if (!arrayin.EvenOdd() || !arrayout.EvenOdd())
      errorCULQCD("Not implemented yet. Aborting...\n");
    if ((arrayin.Mode() == arrayout.Mode() && arrayin.Mode() == Device) &&
        (arrayin.EvenOdd() == arrayout.EvenOdd())) {
      if (arrayin.Size() == arrayout.Size()) {
        dim3 threads(128, 1, 1);
        dim3 blocks = GetBlockDim(threads.x, PARAMS::size);
        kernel_CopyBetweentypes<float, double>
            <<<blocks, threads>>>(arrayin, arrayout, arrayin.Size());
        cudaCheckError("kernel_CopyBetweentypes: Kernel execution failed");
      } else
        errorCULQCD("Cannot copy between arrays with different sizes...\n");
    }
  }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
/**
        @brief Kernel to initialize array with a cold start.
        @param array gauge field
        @param size size of the gauge field including directions
*/
template <ArrayType atype, class Real>
__global__ void kernel_cold_start(complex *array, int size) {
  uint id = INDEX1D();
  if (id < size) {
    GAUGE_SAVE<atype, Real>(array, msun::identity(), id, size);
  }
}
//////////////////////////////////////////////////////////////////
/**
        @brief Initialize array with a cold start in the Device.
        @param array gauge field
*/
template <class Real> void ColdStart(gauge array) {
  dim3 nthreads(128, 1, 1);
  dim3 nblocks = GetBlockDim(nthreads.x, array.Size());
  if (array.Type() == SOA8) {
    COUT << "Array <AOS8> cannot be initialized with a COLD start!"
         << std::endl;
    COUT << "\t-> 8 parameter reconstruction has a singularity at |a1| = 1."
         << std::endl;
    exit(0);
  } else {
    if (array.Type() == SOA)
      kernel_cold_start<SOA, Real>
          <<<nblocks, nthreads>>>(array.GetPtr(), array.Size());
    if (array.Type() == SOA12)
      kernel_cold_start<SOA12, Real>
          <<<nblocks, nthreads>>>(array.GetPtr(), array.Size());
  }
  cudaDeviceSync();
  cudaCheckError("Initialize Array: Kernel execution failed");
  COUT << "Array initialized in the Device!" << std::endl;
}
template void ColdStart(gauges array);
template void ColdStart(gauged array);

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/**
        @brief Kernel to initialize array with a cold start.
        @param array gauge field
        @randstates current state of RNG
        @param size size of the gauge field including directions
        @param rngsize number of the RNG states
*/
template <ArrayType atype, class Real>
__global__ void kernel_hot_start(complex *array, cuRNGState *state, int size,
                                 int rngsize) {
  uint id = INDEX1D();
  if (id < rngsize && id < size) {
    cuRNGState localState = state[id];
    int numberloops = 1;
    if (rngsize < size)
      numberloops = (size + rngsize - 1) / rngsize;
    for (int loop = 0; loop < numberloops; loop++) {
      int idx = id + loop * rngsize;
      if (idx < size) {
        msun U = randomize<Real>(localState);
        reunit_link<Real>(&U);
        GAUGE_SAVE<atype, Real>(array, U, idx, size);
      }
    }
    state[id] = localState;
  }
}

//////////////////////////////////////////////////////////////////
template void HotStart(gauges array, RNG randstates);
template void HotStart(gauged array, RNG randstates);

template <ArrayType atype, class Real>
__global__ void kernel_hot_start1(complex *array, cuRNGState *state, int size,
                                  int rngsize) {
  uint id = INDEX1D();
  if (id < DEVPARAMS::HalfVolume) {
    cuRNGState localState = state[id];
    for (int parity = 0; parity < 2; parity++) {
      int x[NDIMS];
      Index_ND_EO(x, id, parity, DEVPARAMS::Grid);
      for (int i = 0; i < NDIMS; i++)
        x[i] += param_border(i);
      int idxoddbit = Index_ND_NM(x, DEVPARAMS::GridWGhost) >> 1;
      idxoddbit += parity * param_HalfVolumeG();
      int offset = DEVPARAMS::VolumeG * NDIMS;
      for (int mu = 0; mu < NDIMS; mu++) {
        msun U = randomize<Real>(localState);
        reunit_link<Real>(&U);
        GAUGE_SAVE<atype, Real>(array, U, idxoddbit + mu * DEVPARAMS::VolumeG,
                                offset);
      }
    }
    state[id] = localState;
  }
}

/**
        @brief Initialize array with a hot start in the Device.
        @param array gauge field
        @randstates current state of RNG
*/
template <class Real> void HotStart(gauge array, RNG randstates) {
  // if(array.Border() && PARAMS::HalfVolume<=randstates.Size()){
  if (1) {
    dim3 nthreads(128, 1, 1);
    dim3 nblocks = GetBlockDim(nthreads.x, randstates.Size());
    if (array.Type() == SOA)
      kernel_hot_start1<SOA, Real><<<nblocks, nthreads>>>(
          array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
    if (array.Type() == SOA12)
      kernel_hot_start1<SOA12, Real><<<nblocks, nthreads>>>(
          array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
    if (array.Type() == SOA8)
      kernel_hot_start1<SOA8, Real><<<nblocks, nthreads>>>(
          array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
    cudaDeviceSync();
    cudaCheckError("Initialize Array: Kernel execution failed");
    COUT << "Array initialized in the Device, option 1!" << std::endl;

#ifdef MULTI_GPU
    if (numnodes() > 1) {
      CUDA_SAFE_DEVICE_SYNC();
      for (int parity = 0; parity < 2; ++parity)
        for (int mu = 0; mu < 4; ++mu) {
          Exchange_gauge_border_links_gauge(array, mu, parity);
        }
    }
#endif
  } else {
    dim3 nthreads(128, 1, 1);
    dim3 nblocks = GetBlockDim(nthreads.x, randstates.Size());
    if (array.Type() == SOA)
      kernel_hot_start<SOA, Real><<<nblocks, nthreads>>>(
          array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
    if (array.Type() == SOA12)
      kernel_hot_start<SOA12, Real><<<nblocks, nthreads>>>(
          array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
    if (array.Type() == SOA8)
      kernel_hot_start<SOA8, Real><<<nblocks, nthreads>>>(
          array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
    cudaDeviceSync();
    cudaCheckError("Initialize Array: Kernel execution failed");
    COUT << "Array initialized in the Device!" << std::endl;
  }
}

template <ArrayType atype, class Real>
__global__ void kernel_hot_start00(complex *array, cuRNGState *state, int size,
                                   int rngsize) {
  uint id = INDEX1D();
  if (id < rngsize && id < DEVPARAMS::Volume * 4) {
    cuRNGState localState = state[id];
    int numberloops = 1;
    if (rngsize < DEVPARAMS::Volume * 4)
      numberloops = (DEVPARAMS::Volume * 4 + rngsize - 1) / rngsize;
    for (int loop = 0; loop < numberloops; loop++) {
      int idx = id + loop * rngsize;
      if (idx < DEVPARAMS::Volume * 4) {
        msun U = randomize<Real>(localState);
        reunit_link<Real>(&U);
        int newid = idx / 4;
        int nu = idx / DEVPARAMS::Volume;
        newid += nu * DEVPARAMS::tstride * (DEVPARAMS::Grid[3] + 2);
        newid += DEVPARAMS::tstride;
        GAUGE_SAVE<atype, Real>(
            array, U, newid, DEVPARAMS::tstride * (DEVPARAMS::Grid[3] + 2) * 4);
      }
    }
    state[id] = localState;
  }
}

template <class Real> void HotStart00(gauge array, RNG randstates) {
  dim3 nthreads(128, 1, 1);
  dim3 nblocks = GetBlockDim(nthreads.x, randstates.Size());
  if (array.Type() == SOA)
    kernel_hot_start00<SOA, Real><<<nblocks, nthreads>>>(
        array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
  if (array.Type() == SOA12)
    kernel_hot_start00<SOA12, Real><<<nblocks, nthreads>>>(
        array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
  if (array.Type() == SOA8)
    kernel_hot_start00<SOA8, Real><<<nblocks, nthreads>>>(
        array.GetPtr(), randstates.State(), array.Size(), randstates.Size());
  cudaDeviceSync();
  cudaCheckError("Initialize Array: Kernel execution failed");
  COUT << "Array initialized in the Device!" << std::endl;
}
template void HotStart00<float>(gauges array, RNG randstates);
template void HotStart00<double>(gauged array, RNG randstates);

} // namespace CULQCD
