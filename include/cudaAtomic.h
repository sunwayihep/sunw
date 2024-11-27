
#ifndef CUDA_ATOMIC_H
#define CUDA_ATOMIC_H

#include <complex.h>

/*
__inline__ __device__ double atomicAdd(double *addr, double val){
  double old=*addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
            __double_as_longlong(assumed),
            __double_as_longlong(val+assumed)));
  } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );

  return old;
}

__inline__ __device__ CULQCD::complexd atomicAdd(CULQCD::complexd *addr,
CULQCD::complexd val){
    CULQCD::complexd old=*addr;
    old.real() = atomicAdd((double*)addr, val.real());
    old.imag() = atomicAdd((double*)addr+1, val.imag());
    return old;
  }

__inline__ __device__ CULQCD::complexs atomicAdd(CULQCD::complexs *addr,
CULQCD::complexs val){
    CULQCD::complexs old=*addr;
    old.real() = atomicAdd((float*)addr, val.real());
    old.imag() = atomicAdd((float*)addr+1, val.imag());
    return old;
  }


template <typename Real>
struct Summ {
    __host__ __device__ __forceinline__ Real operator()(const Real &a, const
Real &b){
        return a + b;
    }
};
*/

/**
   @brief CUDA double precision version of atomicAd. Reads the word old located
   at the address address in global or shared memory, computes (old + val), and
   stores the result back to memory at the same address. These three operations
   are performed in one atomic transaction. The function returns old.
   @param address memory pointer.
   @param val value to add to current value in memory
   @return function return old
*/ /*
  __device__ double atomicAdd(double* address, double val){ //This is slow!!!!!!
      unsigned long long int* address_as_ull = (unsigned long long int*)address;
      unsigned long long int old = *address_as_ull, assumed;
      do {
          assumed = old;
          old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +
  __longlong_as_double(assumed)));
      } while (assumed != old);
      return __longlong_as_double(old);
  }
  */

namespace CULQCD {

// The default implementation for atomic maximum
template <typename T>
__inline__ __device__ void CudaAtomicMax(T *const address, const T value) {
  atomicMax(address, value);
}

template <>
__inline__ __device__ void CudaAtomicMax(float *const address,
                                         const float value) {
  if (*address >= value) {
    return;
  }

  int *const address_as_i = (int *)address;
  int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) >= value) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(value));
  } while (assumed != old);
}
template <>
__inline__ __device__ void CudaAtomicMax(double *const address,
                                         const double value) {
  if (*address >= value) {
    return;
  }

  unsigned long long int *address_as_i = (unsigned long long int *)address;
  unsigned long long int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= value) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
  } while (assumed != old);
}
template <>
__inline__ __device__ void CudaAtomicMax(complexd *addr, complexd val) {
  CudaAtomicMax((double *)addr, val.real());
  CudaAtomicMax((double *)addr + 1, val.imag());
}
template <>
__inline__ __device__ void CudaAtomicMax(complexs *addr, complexs val) {
  CudaAtomicMax((float *)addr, val.real());
  CudaAtomicMax((float *)addr + 1, val.imag());
}

// The default implementation for atomic minimum
template <typename T>
__inline__ __device__ void CudaAtomicMin(T *const address, const T value) {
  atomicMin(address, value);
}
template <>
__inline__ __device__ void CudaAtomicMin(float *const address,
                                         const float value) {
  if (*address <= value) {
    return;
  }

  int *const address_as_i = (int *)address;
  int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) <= value) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(value));
  } while (assumed != old);
}
template <>
__inline__ __device__ void CudaAtomicMin(double *const address,
                                         const double value) {
  if (*address <= value) {
    return;
  }

  unsigned long long int *address_as_i = (unsigned long long int *)address;
  unsigned long long int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= value) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
  } while (assumed != old);
}
template <>
__inline__ __device__ void CudaAtomicMin(complexd *addr, complexd val) {
  CudaAtomicMin((double *)addr, val.real());
  CudaAtomicMin((double *)addr + 1, val.imag());
}
template <>
__inline__ __device__ void CudaAtomicMin(complexs *addr, complexs val) {
  CudaAtomicMin((float *)addr, val.real());
  CudaAtomicMin((float *)addr + 1, val.imag());
}

// The default implementation for atomic sum/add
__inline__ __device__ float CudaAtomicAdd(float *addr, float val) {
  return atomicAdd(addr, val);
}
__inline__ __device__ double CudaAtomicAdd(double *addr, double val) {
#if __CUDA_ARCH__ < 600
  double old = *addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double(atomicCAS((unsigned long long int *)addr,
                                         __double_as_longlong(assumed),
                                         __double_as_longlong(val + assumed)));
  } while (__double_as_longlong(assumed) != __double_as_longlong(old));

  return old;
#else
  return atomicAdd(addr, val);
#endif
}

__inline__ __device__ complexd CudaAtomicAdd(complexd *addr, complexd val) {
  complexd old = *addr;
  old.real() = CudaAtomicAdd((double *)addr, val.real());
  old.imag() = CudaAtomicAdd((double *)addr + 1, val.imag());
  return old;
}
__inline__ __device__ complexs CudaAtomicAdd(complexs *addr, complexs val) {
  complexs old = *addr;
  old.real() = CudaAtomicAdd((float *)addr, val.real());
  old.imag() = CudaAtomicAdd((float *)addr + 1, val.imag());
  return old;
}

template <typename Real> struct Summ {
  __host__ __device__ __forceinline__ Real operator()(const Real &a,
                                                      const Real &b) {
    return a + b;
  }
};

template <typename T> struct MaxVal {
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) {
    if (a > b)
      return a;
    return b;
  }
};

template <> struct MaxVal<complexs> {
  __host__ __device__ __forceinline__ complexs operator()(const complexs &a,
                                                          const complexs &b) {
    complexs tmp = b;
    if (a.val.x > b.val.x)
      tmp.val.x = a.val.x;
    if (a.val.y > b.val.y)
      tmp.val.y = a.val.y;
    return tmp;
  }
};

template <> struct MaxVal<complexd> {
  __host__ __device__ __forceinline__ complexd operator()(const complexd &a,
                                                          const complexd &b) {
    complexd tmp = b;
    if (a.val.x > b.val.x)
      tmp.val.x = a.val.x;
    if (a.val.y > b.val.y)
      tmp.val.y = a.val.y;
    return tmp;
  }
};

template <typename T> struct MinVal {
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) {
    if (a < b)
      return a;
    return b;
  }
};

template <> struct MinVal<complexs> {
  __host__ __device__ __forceinline__ complexs operator()(const complexs &a,
                                                          const complexs &b) {
    complexs tmp = b;
    if (a.val.x < b.val.x)
      tmp.val.x = a.val.x;
    if (a.val.y < b.val.y)
      tmp.val.y = a.val.y;
    return tmp;
  }
};

template <> struct MinVal<complexd> {
  __host__ __device__ __forceinline__ complexd operator()(const complexd &a,
                                                          const complexd &b) {
    complexd tmp = b;
    if (a.val.x < b.val.x)
      tmp.val.x = a.val.x;
    if (a.val.y < b.val.y)
      tmp.val.y = a.val.y;
    return tmp;
  }
};
} // namespace CULQCD

#endif
