

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>

#include "reduction_kernel.cuh"
#include <complex.h>
#include <cuda_common.h>
#include <reduction.h>

#include <tune.h>

using namespace CULQCD;

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  if (whichKernel < 3) {
    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  }

  if (whichKernel == 6)
    blocks = MIN(maxBlocks, blocks);
}

template <class T>
T reduction(T *array_d, int size, const cudaStream_t &stream) {

  T result;
  int kernel = 6;
  int s = size;
  int maxThreads = 256;
  int maxBlocks = 64;
  int blocks = 0;
  int threads = 0;

  getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
  call_reduce_wstream<T>(s, threads, blocks, kernel, array_d, array_d, stream);

  // sum partial block sums on GPU
  s = blocks;
  while (s > 1) {
    int threads = 0;
    int blocks = 0;
    getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
    call_reduce_wstream<T>(s, threads, blocks, kernel, array_d, array_d,
                           stream);

    if (kernel < 3)
      s = (s + threads - 1) / threads;
    else
      s = (s + (threads * 2 - 1)) / (threads * 2);
  }
  // Copy SUM From GPU to CPU
  CUDA_SAFE_CALL(cudaMemcpyAsync(&result, array_d, sizeof(T),
                                 cudaMemcpyDeviceToHost, stream));
  return result;
}

/*template int
  reduction(int *array_d, int size, const cudaStream_t &stream);
  template int2
  reduction(int2 *array_d, int size, const cudaStream_t &stream);
  template int4
  reduction(int4 *array_d, int size, const cudaStream_t &stream);*/

template float reduction(float *array_d, int size, const cudaStream_t &stream);
template float2 reduction(float2 *array_d, int size,
                          const cudaStream_t &stream);
template float4 reduction(float4 *array_d, int size,
                          const cudaStream_t &stream);

template double reduction(double *array_d, int size,
                          const cudaStream_t &stream);
template double2 reduction(double2 *array_d, int size,
                           const cudaStream_t &stream);
template double4 reduction(double4 *array_d, int size,
                           const cudaStream_t &stream);

template complexs reduction(complexs *array_d, int size,
                            const cudaStream_t &stream);
template complexd reduction(complexd *array_d, int size,
                            const cudaStream_t &stream);

///////////////////// Call Sum reduction
///////////////////////////////////////////////////////////////////////
template <class T> T reduction(T *array_d, int size) {

  return reduction<T>(array_d, size, 0);
}

template float reduction(float *array_d, int size);
template float2 reduction(float2 *array_d, int size);
template float4 reduction(float4 *array_d, int size);

template double reduction(double *array_d, int size);
template double2 reduction(double2 *array_d, int size);
template double4 reduction(double4 *array_d, int size);

template complexs reduction(complexs *array_d, int size);
template complexd reduction(complexd *array_d, int size);