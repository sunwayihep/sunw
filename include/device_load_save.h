
#ifndef DEVICE_SAVE_LOAD_H
#define DEVICE_SAVE_LOAD_H

#include <complex.h>
#include <constants.h>
#include <cuda.h>
#include <gaugearray.h>
#include <matrixsun.h>
#include <reconstruct_12p_8p.h>
#include <texture.h>

namespace CULQCD {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get an element from Device memory,
    it uses texture memory if DEVPARAMS::UseTex is true.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, class Real>
__device__ inline complex ELEM_LOAD(const complex *array, const uint id) {
  if (UseTex)
    return TEXTURE_GAUGE<Real>(id);
  else
    return array[id];
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get a conjugate element from Device memory,
    it uses texture memory if DEVPARAMS::UseTex is true.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, class Real>
__device__ inline complex ELEM_LOAD_CONJ(const complex *array, const uint id) {
  if (UseTex)
    return TEXTURE_GAUGE_CONJ<Real>(id);
  else
    return array[id].conj();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Save SU(Nc) matrix in Device memory. The offset between SU(Nc)
   elements is nx*ny*nz*nt*4
    @param array gauge field
    @param A SU(Nc) matrix to store
    @param id array index
*/
template <ArrayType atype, class Real>
__device__ inline void GAUGE_SAVE(complex *array, const msun A, const uint id) {
  GAUGE_SAVE<atype, Real>(array, A, id, DEVPARAMS::size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Save SU(Nc) matrix from Host/Device memory.
    @param array gauge field
    @param A SU(Nc) matrix to store
    @param id array index
    @param offset stride between SU(Nc) elements in gauge field
*/
template <ArrayType atype, class Real>
__host__ __device__ inline void GAUGE_SAVE(complex *array, const msun A,
                                           const uint id, const uint offset) {
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      array[id + (j + i * NCOLORS) * offset] = A.e[i][j];
#else
  if (atype == SOA) {
    array[id] = A.e[0][0];
    array[id + offset] = A.e[0][1];
    array[id + 2 * offset] = A.e[0][2];
    array[id + 3 * offset] = A.e[1][0];
    array[id + 4 * offset] = A.e[1][1];
    array[id + 5 * offset] = A.e[1][2];
    array[id + 6 * offset] = A.e[2][0];
    array[id + 7 * offset] = A.e[2][1];
    array[id + 8 * offset] = A.e[2][2];
  }
  if (atype == SOA12) {
    array[id] = A.e[0][0];
    array[id + offset] = A.e[0][1];
    array[id + 2 * offset] = A.e[0][2];
    array[id + 3 * offset] = A.e[1][0];
    array[id + 4 * offset] = A.e[1][1];
    array[id + 5 * offset] = A.e[1][2];
  }
  if (atype == SOA12A) {
    array[id] = A.e[0][0];
    array[id + offset] = A.e[0][1];
    array[id + 2 * offset] = A.e[0][2];
    array[id + 3 * offset] = A.e[1][1];
    array[id + 4 * offset] = A.e[1][2];
    array[id + 5 * offset] = A.e[2][2];
  }
  if (atype == SOA8) {
    array[id] = A.e[0][1];
    complex theta;
    theta.real() = A.e[0][0].phase();
    theta.imag() = A.e[2][0].phase();
    array[id + offset] = A.e[0][2];
    array[id + 2 * offset] = A.e[1][0];
    array[id + 3 * offset] = theta;
  }
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get a SU(NC) matrix from Device memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GAUGE_LOAD(const complex *array, const uint id) {
  return GAUGE_LOAD<UseTex, atype, Real>(array, id, DEVPARAMS::size);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get a SU(NC) matrix from Device memory.
    @param array gauge field
    @param id array index
    @param offset stride between SU(Nc) elements in gauge field
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GAUGE_LOAD(const complex *array, const uint id,
                                  const uint offset) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[i][j] =
          ELEM_LOAD<UseTex, Real>(array, id + (j + i * NCOLORS) * offset);
  return A;
#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][0] = ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[1][2] = ELEM_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[2][0] = ELEM_LOAD<UseTex, Real>(array, id + 6 * offset);
    A.e[2][1] = ELEM_LOAD<UseTex, Real>(array, id + 7 * offset);
    A.e[2][2] = ELEM_LOAD<UseTex, Real>(array, id + 8 * offset);
  }
  if (atype == SOA12) {
    A.e[0][0] = ELEM_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][0] = ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[1][2] = ELEM_LOAD<UseTex, Real>(array, id + 5 * offset);
    reconstruct12p<Real>(A);
  }
  if (atype == SOA12A) {
    A.e[0][0] = ELEM_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][1] = ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][2] = ELEM_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[2][2] = ELEM_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[1][0] = complex::make_complex(-A.e[0][1].real(), A.e[0][1].imag());
    A.e[2][0] = complex::make_complex(-A.e[0][2].real(), A.e[0][2].imag());
    A.e[2][1] = complex::make_complex(-A.e[1][2].real(), A.e[1][2].imag());
  }
  if (atype == SOA8) {
    A.e[0][1] = ELEM_LOAD<UseTex, Real>(array, id);
    A.e[0][2] = ELEM_LOAD<UseTex, Real>(array, id + offset);
    A.e[1][0] = ELEM_LOAD<UseTex, Real>(array, id + 2 * offset);
    complex theta = ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);
    reconstruct8p<Real>(A, theta);
  }
  return A;
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get complex conjugate transpose SU(Nc) matrix from Device memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GAUGE_LOAD_DAGGER(const complex *array, const uint id) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[j][i] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + (j + i * NCOLORS) *
                                                               DEVPARAMS::size);
  return A;
#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + DEVPARAMS::size);
    A.e[2][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 2 * DEVPARAMS::size);
    A.e[0][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 3 * DEVPARAMS::size);
    A.e[1][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 4 * DEVPARAMS::size);
    A.e[2][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 5 * DEVPARAMS::size);
    A.e[0][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 6 * DEVPARAMS::size);
    A.e[1][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 7 * DEVPARAMS::size);
    A.e[2][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 8 * DEVPARAMS::size);
  }
  if (atype == SOA12) {
    A.e[0][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + DEVPARAMS::size);
    A.e[2][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 2 * DEVPARAMS::size);
    A.e[0][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 3 * DEVPARAMS::size);
    A.e[1][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 4 * DEVPARAMS::size);
    A.e[2][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 5 * DEVPARAMS::size);
    reconstruct12p_dagger<Real>(A);
  }
  if (atype == SOA12A) {
    A.e[0][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + DEVPARAMS::size);
    A.e[2][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 2 * DEVPARAMS::size);
    A.e[1][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 3 * DEVPARAMS::size);
    A.e[2][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 4 * DEVPARAMS::size);
    A.e[2][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 5 * DEVPARAMS::size);
    A.e[0][1] = complex::make_complex(-A.e[1][0].real(), A.e[1][0].imag());
    A.e[0][2] = complex::make_complex(-A.e[2][0].real(), A.e[2][0].imag());
    A.e[1][2] = complex::make_complex(-A.e[2][1].real(), A.e[2][1].imag());
  }
  if (atype == SOA8) {
    // THIS PART CAN BE OPTIMIZED
    A.e[0][1] = ELEM_LOAD<UseTex, Real>(array, id);
    A.e[0][2] = ELEM_LOAD<UseTex, Real>(array, id + DEVPARAMS::size);
    A.e[1][0] = ELEM_LOAD<UseTex, Real>(array, id + 2 * DEVPARAMS::size);
    complex theta = ELEM_LOAD<UseTex, Real>(array, id + 3 * DEVPARAMS::size);
    reconstruct8p<Real>(A, theta);
    A = A.dagger();
  }
  return A;
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get complex conjugate transpose SU(Nc) matrix from Device memory.
    @param array gauge field
    @param id array index
    @param offset stride between SU(Nc) elements in gauge field
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GAUGE_LOAD_DAGGER(const complex *array, const uint id,
                                         const uint offset) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[j][i] =
          ELEM_LOAD_CONJ<UseTex, Real>(array, id + (j + i * NCOLORS) * offset);
  return A;
#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + offset);
    A.e[2][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 2 * offset);
    A.e[0][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 4 * offset);
    A.e[2][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 5 * offset);
    A.e[0][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 6 * offset);
    A.e[1][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 7 * offset);
    A.e[2][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 8 * offset);
  }
  if (atype == SOA12) {
    A.e[0][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + offset);
    A.e[2][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 2 * offset);
    A.e[0][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 4 * offset);
    A.e[2][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 5 * offset);
    reconstruct12p_dagger<Real>(A);
  }
  if (atype == SOA12A) {
    A.e[0][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + offset);
    A.e[2][0] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 2 * offset);
    A.e[1][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 3 * offset);
    A.e[2][1] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 4 * offset);
    A.e[2][2] = ELEM_LOAD_CONJ<UseTex, Real>(array, id + 5 * offset);
    A.e[0][1] = complex::make_complex(-A.e[1][0].real(), A.e[1][0].imag());
    A.e[0][2] = complex::make_complex(-A.e[2][0].real(), A.e[2][0].imag());
    A.e[1][2] = complex::make_complex(-A.e[2][1].real(), A.e[2][1].imag());
  }
  if (atype == SOA8) {
    // THIS PART CAN BE OPTIMIZED
    A.e[0][1] = ELEM_LOAD<UseTex, Real>(array, id);
    A.e[0][2] = ELEM_LOAD<UseTex, Real>(array, id + offset);
    A.e[1][0] = ELEM_LOAD<UseTex, Real>(array, id + 2 * offset);
    complex theta = ELEM_LOAD<UseTex, Real>(array, id + 3 * offset);
    reconstruct8p<Real>(A, theta);
    A = A.dagger();
  }
  return A;
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get a Delta(x) element from Device memory,
    it uses texture memory if DEVPARAMS::UseTex is true.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, class Real>
__device__ inline complex ELEM_DELTA_LOAD(const complex *array, const uint id) {
  if (UseTex)
    return TEXTURE_DELTA<Real>(id);
  else
    return array[id];
}
/**
    @brief Get a Delta(x) conjugate element from Device memory,
    it uses texture memory if DEVPARAMS::UseTex is true.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, class Real>
__device__ inline complex ELEM_DELTA_LOAD_CONJ(const complex *array,
                                               const uint id) {
  if (UseTex)
    return TEXTURE_DELTA_CONJ<Real>(id);
  else
    return array[id].conj();
}

/**
    @brief Load Delta(x) SU(Nc) matrix from Device memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun DELTA_LOAD(const complex *array, const uint id) {
  return DELTA_LOAD<UseTex, atype, Real>(array, id, DEVPARAMS::Volume);
}
/**
    @brief Load complex conjugate transpose Delta(x) SU(Nc) matrix from Device
   memory.
    @param array gauge field
    @param id array index
    @param offset stride between SU(Nc) elements in gauge field
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun DELTA_LOAD_DAGGER(const complex *array, const uint id) {
  return DELTA_LOAD_DAGGER<UseTex, atype, Real>(array, id, DEVPARAMS::Volume);
}

/**
    @brief Load Delta(x) SU(Nc) matrix from Device memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun DELTA_LOAD(const complex *array, const uint id,
                                  const uint offset) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[i][j] =
          ELEM_DELTA_LOAD<UseTex, Real>(array, id + (j + i * NCOLORS) * offset);
  return A;
#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_DELTA_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][0] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[1][2] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[2][0] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 6 * offset);
    A.e[2][1] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 7 * offset);
    A.e[2][2] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 8 * offset);
  }
  if (atype == SOA12A) {
    A.e[0][0] = ELEM_DELTA_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][1] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][2] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[2][2] = ELEM_DELTA_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[1][0] = complex::make_complex(-A.e[0][1].real(), A.e[0][1].imag());
    A.e[2][0] = complex::make_complex(-A.e[0][2].real(), A.e[0][2].imag());
    A.e[2][1] = complex::make_complex(-A.e[1][2].real(), A.e[1][2].imag());
  }
  if (atype == SOA8 || atype == SOA12) {
    //        #error Function not available for delta....
  }
  return A;
#endif
}

template <ArrayType atype, class Real>
__host__ __device__ inline void DELTA_SAVE(complex *array, msun A, uint id,
                                           uint offset) {
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      array[id + (j + i * NCOLORS) * offset] = A.e[i][j];
#else
  if (atype == SOA) {
    array[id] = A.e[0][0];
    array[id + offset] = A.e[0][1];
    array[id + 2 * offset] = A.e[0][2];
    array[id + 3 * offset] = A.e[1][0];
    array[id + 4 * offset] = A.e[1][1];
    array[id + 5 * offset] = A.e[1][2];
    array[id + 6 * offset] = A.e[2][0];
    array[id + 7 * offset] = A.e[2][1];
    array[id + 8 * offset] = A.e[2][2];
  }
  if (atype == SOA12A) {
    array[id] = A.e[0][0];
    array[id + offset] = A.e[0][1];
    array[id + 2 * offset] = A.e[0][2];
    array[id + 3 * offset] = A.e[1][1];
    array[id + 4 * offset] = A.e[1][2];
    array[id + 5 * offset] = A.e[2][2];
  }
  if (atype == SOA8 || atype == SOA12) {
    //       #error Function not available for delta....
  }
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get a g(x) element from Device memory,
    it uses texture memory if DEVPARAMS::UseTex is true.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, class Real>
__device__ inline complex ELEM_GX_LOAD(const complex *array, const uint id) {
  if (UseTex)
    return TEXTURE_GX<Real>(id);
  else
    return array[id];
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Get a g(x) conjugate element from Device memory,
    it uses texture memory if DEVPARAMS::UseTex is true.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, class Real>
__device__ inline complex ELEM_GX_LOAD_CONJ(const complex *array,
                                            const uint id) {
  if (UseTex)
    return TEXTURE_GX_CONJ<Real>(id);
  else
    return array[id].conj();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Load g(x) SU(Nc) matrix from Device memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GX_LOAD(const complex *array, const uint id) {
  return GX_LOAD<UseTex, atype, Real>(array, id, DEVPARAMS::Volume);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Load complex conjugate transpose g(x) SU(Nc) matrix from Device
   memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GX_LOAD_DAGGER(const complex *array, const uint id) {
  return GX_LOAD_DAGGER<UseTex, atype, Real>(array, id, DEVPARAMS::Volume);
}

/**
    @brief Load g(x) SU(Nc) matrix from Device memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GX_LOAD(const complex *array, const uint id,
                               const uint offset) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[i][j] =
          ELEM_GX_LOAD<UseTex, Real>(array, id + (j + i * NCOLORS) * offset);
  return A;

#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_GX_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_GX_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][0] = ELEM_GX_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_GX_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[1][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[2][0] = ELEM_GX_LOAD<UseTex, Real>(array, id + 6 * offset);
    A.e[2][1] = ELEM_GX_LOAD<UseTex, Real>(array, id + 7 * offset);
    A.e[2][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + 8 * offset);
  }
  if (atype == SOA12) {
    A.e[0][0] = ELEM_GX_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_GX_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][0] = ELEM_GX_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_GX_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[1][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + 5 * offset);
    reconstruct12p<Real>(A);
  }
  if (atype == SOA8) {
    A.e[0][1] = ELEM_GX_LOAD<UseTex, Real>(array, id);
    A.e[0][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + offset);
    A.e[1][0] = ELEM_GX_LOAD<UseTex, Real>(array, id + 2 * offset);
    complex theta = ELEM_GX_LOAD<UseTex, Real>(array, id + 3 * offset);
    reconstruct8p<Real>(A, theta);
  }
  return A;
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    @brief Load complex conjugate transpose g(x) SU(Nc) matrix from Device
   memory.
    @param array gauge field
    @param id array index
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun GX_LOAD_DAGGER(const complex *array, const uint id,
                                      const uint offset) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[j][i] = ELEM_GX_LOAD_CONJ<UseTex, Real>(
          array, id + (j + i * NCOLORS) * offset);
  return A;
#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + offset);
    A.e[2][0] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 2 * offset);
    A.e[0][1] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 4 * offset);
    A.e[2][1] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 5 * offset);
    A.e[0][2] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 6 * offset);
    A.e[1][2] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 7 * offset);
    A.e[2][2] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 8 * offset);
  }
  if (atype == SOA12) {
    A.e[0][0] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id);
    A.e[1][0] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + offset);
    A.e[2][0] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 2 * offset);
    A.e[0][1] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 4 * offset);
    A.e[2][1] = ELEM_GX_LOAD_CONJ<UseTex, Real>(array, id + 5 * offset);
    reconstruct12p_dagger<Real>(A);
  }
  if (atype == SOA8) {
    // THIS PART CAN BE OPTIMIZED
    A.e[0][1] = ELEM_GX_LOAD<UseTex, Real>(array, id);
    A.e[0][2] = ELEM_GX_LOAD<UseTex, Real>(array, id + offset);
    A.e[1][0] = ELEM_GX_LOAD<UseTex, Real>(array, id + 2 * offset);
    complex theta = ELEM_GX_LOAD<UseTex, Real>(array, id + 3 * offset);
    reconstruct8p<Real>(A, theta);
    A = A.dagger();
  }
  return A;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, class Real>
__device__ inline complex ELEM_LAMBDA_LOAD(const complex *array,
                                           const uint id) {
  if (UseTex)
    return TEXTURE_LAMBDA<Real>(id);
  else
    return array[id];
}

template <bool UseTex, ArrayType atype, class Real>
__device__ inline msun LAMBDA_LOAD(const complex *array, const uint id,
                                   const uint offset) {
  msun A;
#if (NCOLORS > 3) || (NCOLORS == 2)
#pragma unroll
  for (int i = 0; i < NCOLORS; i++)
#pragma unroll
    for (int j = 0; j < NCOLORS; j++)
      A.e[i][j] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + (j + i * NCOLORS) *
                                                                 offset);
  return A;

#else
  if (atype == SOA) {
    A.e[0][0] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][0] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][1] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[1][2] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[2][0] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 6 * offset);
    A.e[2][1] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 7 * offset);
    A.e[2][2] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 8 * offset);
  }
  if (atype == SOA12A) {
    A.e[0][0] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id);
    A.e[0][1] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + offset);
    A.e[0][2] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 2 * offset);
    A.e[1][1] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 3 * offset);
    A.e[1][2] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 4 * offset);
    A.e[2][2] = ELEM_LAMBDA_LOAD<UseTex, Real>(array, id + 5 * offset);
    A.e[1][0] = complex::make_complex(-A.e[0][1].real(), A.e[0][1].imag());
    A.e[2][0] = complex::make_complex(-A.e[0][2].real(), A.e[0][2].imag());
    A.e[2][1] = complex::make_complex(-A.e[1][2].real(), A.e[1][2].imag());
  }
  if (atype == SOA8 || atype == SOA12) {
  }
  return A;
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace CULQCD

#endif // #ifndef DEVICE_SAVE_LOAD_H
