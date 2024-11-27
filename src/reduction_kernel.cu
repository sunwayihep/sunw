/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
  Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <complex.h>
#include <cuda_common.h>
#include <stdio.h>
#include <vector_types.h>

#include "reduce_T2_op.cuh"
#include <sharedmemtypes.h>

using namespace CULQCD;

// original
/*
  #ifdef __DEVICE_EMULATION__
  #define EMUSYNC __syncthreads()
  #else
  #define EMUSYNC
  #endif
*/
// reason -> cannot use volatile with float2, double2...
// if not use __syncthreads() it gives wrong results in fermi gpu, not in gtx295
// gpu
#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC __syncthreads()
#endif

/*
  This version adds multiple elements per thread sequentially.  This reduces the
  overall cost of the algorithm while keeping the work complexity O(n) and the
  step complexity O(log n). (Brent's Theorem optimization)

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = zero<T>();

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];
    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i + blockSize];
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = mySum = mySum + sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = mySum = mySum + sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    T *smem = sdata;
    if (blockSize >= 64) {
      smem[tid] = mySum = mySum + smem[tid + 32];
      EMUSYNC;
    }
    if (blockSize >= 32) {
      smem[tid] = mySum = mySum + smem[tid + 16];
      EMUSYNC;
    }
    if (blockSize >= 16) {
      smem[tid] = mySum = mySum + smem[tid + 8];
      EMUSYNC;
    }
    if (blockSize >= 8) {
      smem[tid] = mySum = mySum + smem[tid + 4];
      EMUSYNC;
    }
    if (blockSize >= 4) {
      smem[tid] = mySum = mySum + smem[tid + 2];
      EMUSYNC;
    }
    if (blockSize >= 2) {
      smem[tid] = mySum = mySum + smem[tid + 1];
      EMUSYNC;
    }
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

extern "C" bool isPow2(unsigned int x);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void call_reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
                 T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size)) {
    switch (threads) {
    case 512:
      reduce6<T, 512, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 256:
      reduce6<T, 256, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 128:
      reduce6<T, 128, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 64:
      reduce6<T, 64, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 32:
      reduce6<T, 32, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 16:
      reduce6<T, 16, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 8:
      reduce6<T, 8, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 4:
      reduce6<T, 4, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 2:
      reduce6<T, 2, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 1:
      reduce6<T, 1, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    }
  } else {
    switch (threads) {
    case 512:
      reduce6<T, 512, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 256:
      reduce6<T, 256, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 128:
      reduce6<T, 128, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 64:
      reduce6<T, 64, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 32:
      reduce6<T, 32, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 16:
      reduce6<T, 16, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 8:
      reduce6<T, 8, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 4:
      reduce6<T, 4, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 2:
      reduce6<T, 2, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 1:
      reduce6<T, 1, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    }
  }
}

/*template void
  call_reduce<int>(int size, int threads, int blocks,
  int whichKernel, int *d_idata, int *d_odata);
  template void
  call_reduce<int2>(int size, int threads, int blocks,
  int whichKernel, int2 *d_idata, int2 *d_odata);
  template void
  call_reduce<int4>(int size, int threads, int blocks,
  int whichKernel, int4 *d_idata, int4 *d_odata);*/

template void call_reduce<float>(int size, int threads, int blocks,
                                 int whichKernel, float *d_idata,
                                 float *d_odata);
template void call_reduce<float2>(int size, int threads, int blocks,
                                  int whichKernel, float2 *d_idata,
                                  float2 *d_odata);
template void call_reduce<float4>(int size, int threads, int blocks,
                                  int whichKernel, float4 *d_idata,
                                  float4 *d_odata);

template void call_reduce<double>(int size, int threads, int blocks,
                                  int whichKernel, double *d_idata,
                                  double *d_odata);
template void call_reduce<double2>(int size, int threads, int blocks,
                                   int whichKernel, double2 *d_idata,
                                   double2 *d_odata);
template void call_reduce<double4>(int size, int threads, int blocks,
                                   int whichKernel, double4 *d_idata,
                                   double4 *d_odata);

template void call_reduce<complexs>(int size, int threads, int blocks,
                                    int whichKernel, complexs *d_idata,
                                    complexs *d_odata);
template void call_reduce<complexd>(int size, int threads, int blocks,
                                    int whichKernel, complexd *d_idata,
                                    complexd *d_odata);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void call_reduce_wstream(int size, int threads, int blocks, int whichKernel,
                         T *d_idata, T *d_odata, const cudaStream_t &stream) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size)) {
    switch (threads) {
    case 512:
      reduce6<T, 512, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 256:
      reduce6<T, 256, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 128:
      reduce6<T, 128, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 64:
      reduce6<T, 64, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 32:
      reduce6<T, 32, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 16:
      reduce6<T, 16, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 8:
      reduce6<T, 8, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 4:
      reduce6<T, 4, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 2:
      reduce6<T, 2, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 1:
      reduce6<T, 1, true>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    }
  } else {
    switch (threads) {
    case 512:
      reduce6<T, 512, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 256:
      reduce6<T, 256, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 128:
      reduce6<T, 128, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 64:
      reduce6<T, 64, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 32:
      reduce6<T, 32, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 16:
      reduce6<T, 16, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 8:
      reduce6<T, 8, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 4:
      reduce6<T, 4, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 2:
      reduce6<T, 2, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    case 1:
      reduce6<T, 1, false>
          <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
      CUT_CHECK_ERROR("Reduce: Kernel execution failed");
      CUDA_SAFE_DEVICE_SYNC();
      break;
    }
  }
}

/*template void
  call_reduce_wstream<int>(int size, int threads, int blocks,
  int whichKernel, int *d_idata, int *d_odata, const cudaStream_t &stream);
  template void
  call_reduce_wstream<int2>(int size, int threads, int blocks,
  int whichKernel, int2 *d_idata, int2 *d_odata, const cudaStream_t &stream);
  template void
  call_reduce_wstream<int4>(int size, int threads, int blocks,
  int whichKernel, int4 *d_idata, int4 *d_odata, const cudaStream_t &stream);*/

template void call_reduce_wstream<float>(int size, int threads, int blocks,
                                         int whichKernel, float *d_idata,
                                         float *d_odata,
                                         const cudaStream_t &stream);
template void call_reduce_wstream<float2>(int size, int threads, int blocks,
                                          int whichKernel, float2 *d_idata,
                                          float2 *d_odata,
                                          const cudaStream_t &stream);
template void call_reduce_wstream<float4>(int size, int threads, int blocks,
                                          int whichKernel, float4 *d_idata,
                                          float4 *d_odata,
                                          const cudaStream_t &stream);

template void call_reduce_wstream<double>(int size, int threads, int blocks,
                                          int whichKernel, double *d_idata,
                                          double *d_odata,
                                          const cudaStream_t &stream);
template void call_reduce_wstream<double2>(int size, int threads, int blocks,
                                           int whichKernel, double2 *d_idata,
                                           double2 *d_odata,
                                           const cudaStream_t &stream);
template void call_reduce_wstream<double4>(int size, int threads, int blocks,
                                           int whichKernel, double4 *d_idata,
                                           double4 *d_odata,
                                           const cudaStream_t &stream);

template void call_reduce_wstream<complexs>(int size, int threads, int blocks,
                                            int whichKernel, complexs *d_idata,
                                            complexs *d_odata,
                                            const cudaStream_t &stream);
template void call_reduce_wstream<complexd>(int size, int threads, int blocks,
                                            int whichKernel, complexd *d_idata,
                                            complexd *d_odata,
                                            const cudaStream_t &stream);

#endif // #ifndef _REDUCE_KERNEL_H_
