
#ifndef HOSTDEVICE_INDEX_H
#define HOSTDEVICE_INDEX_H

#include <constants.h>
#include <vector_types.h>

namespace CULQCD {

////////////////////////////////////////////////////////////////////
///////////   Device Functions  ////////////////////////////////////
////////////////////////////////////////////////////////////////////

// From normal to normal lattice index
// for N-dimensional lattice
__host__ __device__ inline void Index_ND_NM(const int id, int x[NDIMS]) {
  int temp = id;
  for (int i = 0; i < NDIMS; i++) {
    x[i] = temp % param_Grid(i);
    temp /= param_Grid(i);
  }
}

__host__ __device__ inline void Index_ND_NM(const int id, int x[NDIMS],
                                            const int X[NDIMS]) {
  int temp = id;
  for (int i = 0; i < NDIMS; i++) {
    x[i] = temp % X[i];
    temp /= X[i];
  }
}

__host__ __device__ inline int Index_ND_NM(const int y[NDIMS]) {
  int index = 0;
  int factor = 1;
  for (int i = 0; i < NDIMS; i++) {
    index += y[i] * factor;
    factor *= param_Grid(i);
  }

  return index;
}

__host__ __device__ inline int Index_ND_NM(const int x[NDIMS],
                                           const int X[NDIMS]) {
  int index = 0;
  int factor = 1;
  for (int i = 0; i < NDIMS; i++) {
    index += x[i] * factor;
    factor *= X[i];
  }

  return index;
}

__host__ __device__ inline int Index_ND_Neig_NM(const int id, const int mu,
                                                const int lmu, const int nu,
                                                const int lnu) {
  int x[NDIMS];
  Index_ND_NM(id, x);
  x[mu] = (x[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  x[nu] = (x[nu] + lnu + param_Grid(nu)) % param_Grid(nu);

  return Index_ND_NM(x);
}

__host__ __device__ inline int Index_ND_Neig_NM(const int id, const int mu,
                                                const int lmu) {
  int x[NDIMS];
  Index_ND_NM(id, x);
  x[mu] = (x[mu] + lmu + param_Grid(mu)) % param_Grid(mu);

  return Index_ND_NM(x);
}

__host__ __device__ inline int Index_ND_Neig_NM(const int x[NDIMS],
                                                const int mu, const int lmu) {
  int y[NDIMS];
  for (int i = 0; i < NDIMS; i++)
    y[i] = x[i];
  y[mu] = (x[mu] + lmu + param_Grid(mu)) % param_Grid(mu);

  return Index_ND_NM(y);
}

__host__ __device__ inline int Index_ND_Neig_NM(const int x[NDIMS],
                                                int dx[NDIMS]) {
  int y[NDIMS];
  for (int i = 0; i < NDIMS; i++)
    y[i] = (x[i] + dx[i] + param_Grid(i)) % param_Grid(i);

  return Index_ND_NM(y);
}

__host__ __device__ inline void Index_ND_Neig_NM(int y[NDIMS], int x[NDIMS],
                                                 int mu, int lmu) {
  for (int i = 0; i < NDIMS; i++)
    y[i] = x[i];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
}

__host__ __device__ inline void
Index_ND_Neig_NM(int y[NDIMS], int x[NDIMS], int mu, int lmu, int nu, int lnu) {
  for (int i = 0; i < NDIMS; i++)
    y[i] = x[i];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  y[nu] = (y[nu] + lnu + param_Grid(nu)) % param_Grid(nu);
}

// Spatial part of the NDIMS-dim lattice, i.e. (NDIMS -1) dimension
__host__ __device__ inline void Index_NDs_NM(const int id, int x[NDIMS - 1]) {
  int temp = id;
  for (int i = 0; i < NDIMS - 1; i++) {
    x[i] = temp % param_Grid(i);
    temp /= param_Grid(i);
  }
}

__host__ __device__ inline int Index_NDs_NM(const int y[NDIMS - 1]) {
  int index = 0;
  int factor = 1;
  for (int i = 0; i < NDIMS - 1; i++) {
    index += y[i] * factor;
    factor *= param_Grid(i);
  }

  return index;
}

__host__ __device__ inline int Index_NDs_Neig_NM(const int id, int mu, int r) {
  int x[NDIMS - 1];
  Index_NDs_NM(id, x);
  x[mu] = (x[mu] + r + param_Grid(mu)) % param_Grid(mu);

  return Index_NDs_NM(x);
}

__host__ __device__ inline int Index_NDs_Neig_NM(const int x[NDIMS - 1],
                                                 const int dx[NDIMS - 1]) {
  int y[NDIMS - 1];
  for (int i = 0; i < NDIMS - 1; i++)
    y[i] = (x[i] + dx[i] + param_Grid(i)) % param_Grid(i);

  return Index_NDs_NM(y);
}

__host__ __device__ inline int Index_NDs_Neig_NM(const int x[NDIMS - 1], int mu,
                                                 int lmu, int nu, int lnu) {
  int y[NDIMS];
  for (int dir = 0; dir < NDIMS - 1; dir++)
    y[dir] = x[dir];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  y[nu] = (y[nu] + lnu + param_Grid(nu)) % param_Grid(nu);

  return Index_NDs_NM(y);
}

__host__ __device__ inline int Index_NDs_Neig_NM(const int x[NDIMS - 1], int mu,
                                                 int lmu) {
  int y[NDIMS - 1];
  for (int dir = 0; dir < NDIMS - 1; dir++)
    y[dir] = x[dir];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);

  return Index_NDs_NM(y);
}

// From EO to EO lattice index

__host__ __device__ inline void Index_ND_EO(int x[NDIMS], const int id,
                                            const int oddbit) {
  int factor = id / (param_Grid(0) / 2);
  for (int i = 1; i < NDIMS; i++) {
    int factor1 = factor / param_Grid(i);
    x[i] = factor - factor1 * param_Grid(i);
    factor = factor1;
  }
  int sum = 0;
  for (int i = 1; i < NDIMS; i++)
    sum += x[i];
  int xodd = (sum + oddbit) & 1;
  x[0] = (id * 2 + xodd) - id / (param_Grid(0) / 2) * param_Grid(0);
}

__host__ __device__ inline void
Index_ND_EO(int x[NDIMS], const int id, const int oddbit, const int X[NDIMS]) {
  int factor = id / (X[0] / 2);
  for (int i = 1; i < NDIMS; i++) {
    int factor1 = factor / X[i];
    x[i] = factor - factor1 * X[i];
    factor = factor1;
  }
  int sum = 0;
  for (int i = 1; i < NDIMS; i++)
    sum += x[i];
  int xodd = (sum + oddbit) & 1;
  x[0] = (id * 2 + xodd) - id / (X[0] / 2) * X[0];
}

__host__ __device__ inline int
Index_ND_Neig_EO(const int x[NDIMS], const int dx[NDIMS], const int X[NDIMS]) {
  int y[NDIMS];
  for (int i = 0; i < NDIMS; i++)
    y[i] = (x[i] + dx[i] + X[i]) % X[i];

  int index = 0;
  int factor = 1;
  for (int i = 0; i < NDIMS; i++) {
    index += y[i] * factor;
    factor *= X[i];
  }

  return index >> 1;
}

/**
        @brief U(id + lmu * e_mu), retrieves the neighbor in evenodd lattice
   index
*/
__host__ __device__ inline int Index_ND_Neig_EO(const int id, int oddbit,
                                                int mu, int lmu) {
  int x[NDIMS];
  Index_ND_EO(x, id, oddbit);
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] += param_border(i);
#endif
  x[mu] = (x[mu] + lmu + param_GridG(mu)) % param_GridG(mu);

  int pos = Index_ND_NM(x) / 2;
  int sum_x = 0;
  for (int i = 0; i < NDIMS; i++)
    sum_x += x[i];
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * param_HalfVolumeG();

  return pos;
}

/**
        @brief U(id + lmu * e_mu + lnu * e_nu), retrieves the neighbor in
   evenodd lattice index
*/
__host__ __device__ inline int
Index_ND_Neig_EO(const int id, int oddbit, int mu, int lmu, int nu, int lnu) {
  int x[NDIMS];
  Index_ND_EO(x, id, oddbit);
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] += param_border(i);
#endif
  x[mu] = (x[mu] + lmu + param_GridG(mu)) % param_GridG(mu);
  x[nu] = (x[nu] + lnu + param_GridG(nu)) % param_GridG(nu);

  int pos = Index_ND_NM(x) / 2;
  int sum_x = 0;
  for (int i = 0; i < NDIMS; i++)
    sum_x += x[i];
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * param_HalfVolumeG();
  return pos;
}

/**
        @brief U(id + lmu * e_mu), retrieves the neighbor in evenodd lattice
   index
*/
__host__ __device__ inline int Index_ND_Neig_EO(const int y[NDIMS], int oddbit,
                                                int mu, int lmu) {
  int x[NDIMS];
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] = y[i] + param_border(i);
#else
  for (int i = 0; i < NDIMS; i++)
    x[i] = y[i];
#endif
  x[mu] = (x[mu] + lmu + param_GridG(mu)) % param_GridG(mu);
  int pos = Index_ND_NM(x) / 2;
  int sum_x = 0;
  for (int i = 0; i < NDIMS; i++)
    sum_x += x[i];
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * param_HalfVolumeG();

  return pos;
}

/**
        @brief U(id + lmu * e_mu + lnu * e_nu), retrieves the neighbor in
   evenodd lattice index
*/
__host__ __device__ inline int Index_ND_Neig_EO(const int y[NDIMS], int oddbit,
                                                int mu, int lmu, int nu,
                                                int lnu) {
  int x[NDIMS];
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] = y[i] + param_border(i);
#else
  for (int i = 0; i < NDIMS; i++)
    x[i] = y[i];
#endif
  x[mu] = (x[mu] + lmu + param_GridG(mu)) % param_GridG(mu);
  x[nu] = (x[nu] + lnu + param_GridG(nu)) % param_GridG(nu);
  int pos = Index_ND_NM(x) / 2;
  int sum_x = 0;
  for (int i = 0; i < NDIMS; i++)
    sum_x += x[i];
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * param_HalfVolumeG();

  return pos;
}

/**
        @brief U(id + e_mu), retrieves the neighbor in evenodd lattice index
*/
__host__ __device__ inline int
Index_ND_Neig_EO_PlusOne(const int id, const int oddbit, const int mu) {
  int x[NDIMS];
  Index_ND_EO(x, id, oddbit);
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] += param_border(i);
#endif
  x[mu] = (x[mu] + 1) % param_GridG(mu);
  int idx = (Index_ND_NM(x) >> 1);
  idx += (1 - oddbit) * param_HalfVolumeG();

  return idx;
}

/**
        @brief U(id - e_mu), retrieves the neighbor in evenodd lattice index
*/
__host__ __device__ inline int
Index_ND_Neig_EO_MinusOne(const int id, const int oddbit, const int mu) {
  int x[NDIMS];
  Index_ND_EO(x, id, oddbit);
#ifdef MULTI_GPU
  for (int i = 0; i < NDIMS; i++)
    x[i] += param_border(i);
#endif
  x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
  int idx = (Index_ND_NM(x) >> 1);
  idx += (1 - oddbit) * param_HalfVolumeG();

  return idx;
}

#ifdef __CUDACC__
// #define __float2int_rd(a) a
#define _mul(a, b) ((a) * (b))

/**
        @brief Returns a int3 the global 3D thread index for a 3D thread block.
*/
__forceinline__ int4 __device__ INDEX4D() {
  int4 id;
  int ij = _mul(blockIdx.x, blockDim.x) + threadIdx.x;
  id.y = __float2int_rd(ij / param_Grid(0));
  id.x = ij - id.y * param_Grid(0);
  id.z = _mul(blockIdx.y, blockDim.y) + threadIdx.y;
  id.w = _mul(blockIdx.z, blockDim.z) + threadIdx.z;
  return id;
}
/**
        @brief Returns a int3 the global 3D thread index for a 3D thread block.
*/
__forceinline__ int3 __device__ INDEX3D() {
  int3 id;
  id.x = _mul(blockIdx.x, blockDim.x) + threadIdx.x;
  id.y = _mul(blockIdx.y, blockDim.y) + threadIdx.y;
  id.z = _mul(blockIdx.z, blockDim.z) + threadIdx.z;
  return id;
}
/**
        @brief Returns the global thread index for 1D thread block.
        In Fermi architecture the 1D grid size 	is limited to 65535 blocks
        and therefore when this is insufficient a 2D grid size is set.
*/
__forceinline__ uint __device__ INDEX1D() {
#if (__CUDA_ARCH__ >= 300)
  return blockIdx.x * blockDim.x + threadIdx.x;
#else
  uint id = gridDim.x * blockIdx.y + blockIdx.x;
  return blockDim.x * id + threadIdx.x;
#endif
}
#endif
} // namespace CULQCD

#endif
