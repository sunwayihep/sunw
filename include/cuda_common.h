

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#ifdef _WIN32
#pragma warning(disable : 4996) // disable deprecated warning
#endif

#include <stdio.h>
#include <stdlib.h>

namespace CULQCD {

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSDIVUP(a, b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////
//! Macros

// Depending on whether we're running inside the CUDA compiler, define the
// __host_
// and __device__ intrinsics, otherwise just make the functions static to
// prevent
// linkage issues (duplicate symbols and such)
#ifdef __CUDACC__
//#define HOST __host__
#define DEVICE __device__ inline
#define HOSTDEVICE __host__ __device__ inline
#define M_HOST __host__
#define M_HOSTDEVICE __host__ __device__ inline
#else
//#define HOST static inline
#define DEVICE static inline
#define HOSTDEVICE static inline
#define M_HOST inline // note there is no static here
#define M_HOSTDEVICE                                                           \
  inline // (static has a different meaning for class member functions)
#endif
/*
  #define HOST static inline
  #define DEVICE static inline __device__
  #define HOSTDEVICE static inline __host__ __device__
  #define M_HOST inline      // note there is no static here
  #define M_HOSTDEVICE inline __host__ __device__ // (static has a different
  meaning for class member functions)
*/

// Struct alignment is handled differently between the CUDA compiler and other
// compilers (e.g. GCC, MS Visual C++ .NET)
#ifdef __CUDACC__
#define ALIGN(x) __align__(x)
#else
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
// Visual C++ .NET and later
#define ALIGN(x) __declspec(align(x))
#else
#if defined(__GNUC__)
// GCC
#define ALIGN(x) __attribute__((aligned(x)))
#else
// all other compilers
#define ALIGN(x)
#endif
#endif
#endif

#if !defined(__DEVICE_EMULATION__) || (defined(_MSC_VER) && (_MSC_VER >= 1300))
#define REF(x) &x
#define ARRAYREF(x, y) (&x)[y]
#define PTR(x) *x
#else
#define REF(x) x
#define ARRAYREF(x, y) x[y]
#define PTR(x) *x
#endif

//#if CUDART_VERSION >= 4000
#define CUT_DEVICE_SYNCHRONIZE() cudaDeviceSynchronize();
#define CUT_DEVICE_RESET() cudaDeviceReset();
//#else
//#define CUT_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
//#define CUT_DEVICE_RESET( )   cudaThreadExit();
//#endif

/*-------------------------------------------------------------------------------*/
#define OUTPUT_RESULT(result)                                                  \
  switch (result) {                                                            \
  case cudaSuccess:                                                            \
    printf(" => cudaSuccess\n");                                               \
    break;                                                                     \
  case cudaErrorInvalidValue:                                                  \
    printf(" => cudaErrorInvalidValue\n");                                     \
    break;                                                                     \
  case cudaErrorInvalidSymbol:                                                 \
    printf(" => cudaErrorInvalidSymbol\n");                                    \
    break;                                                                     \
  case cudaErrorInvalidDevicePointer:                                          \
    printf(" => cudaErrorInvalidDevicePointer\n");                             \
    break;                                                                     \
  case cudaErrorInvalidMemcpyDirection:                                        \
    printf(" => cudaErrorInvalidMemcpyDirection\n");                           \
    break;                                                                     \
  default:                                                                     \
    printf(" => unknown\n");                                                   \
    break;                                                                     \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
#define CU_SAFE_CALL_NO_SYNC(call)                                             \
  {                                                                            \
    CUresult err = call;                                                       \
    if (CUDA_SUCCESS != err) {                                                 \
      fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n", err,  \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CU_SAFE_CALL(call) CU_SAFE_CALL_NO_SYNC(call);
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
#define CU_SAFE_CTX_SYNC()                                                     \
  {                                                                            \
    CUresult err = cuCtxSynchronize();                                         \
    if (CUDA_SUCCESS != err) {                                                 \
      fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n", err,  \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
#define CUDA_SAFE_CALL_NO_SYNC(call)                                           \
  {                                                                            \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_SAFE_CALL(call) CUDA_SAFE_CALL_NO_SYNC(call);
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
#define CUDA_SAFE_THREAD_SYNC()                                                \
  {                                                                            \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
    }                                                                          \
  }
#define CUDA_SAFE_DEVICE_SYNC()                                                \
  {                                                                            \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
    }                                                                          \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
#define CUFFT_SAFE_CALL(call)                                                  \
  {                                                                            \
    cufftResult err = call;                                                    \
    if (CUFFT_SUCCESS != err) {                                                \
      fprintf(stderr, "CUFFT error in file '%s' in line %i.\n", __FILE__,      \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
/*-------------------------------------------------------------------------------*/

#define CUT_SAFE_CALL(call)                                                    \
  if (CUTTrue != call) {                                                       \
    fprintf(stderr, "Cut error in file '%s' in line %i.\n", __FILE__,          \
            __LINE__);                                                         \
    exit(EXIT_FAILURE);                                                        \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
//! Check for CUDA error
#ifdef _DEBUG
#define CUT_CHECK_ERROR(errorMessage)                                          \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    err = CUT_DEVICE_SYNCHRONIZE();                                            \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#else
#define CUT_CHECK_ERROR(errorMessage)                                          \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#endif
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
//! Check for malloc error
#define CUT_SAFE_MALLOC(mallocCall)                                            \
  {                                                                            \
    if (!(mallocCall)) {                                                       \
      fprintf(stderr, "Host malloc failure in file '%s' in line %i\n",         \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }                                                                            \
  while (0)                                                                    \
    ;
/*-------------------------------------------------------------------------------*/

//! Check if conditon is true (flexible assert)
#define CUT_CONDITION(val)                                                     \
  if (CUTFalse == cutCheckCondition(val, __FILE__, __LINE__)) {                \
    exit(EXIT_FAILURE);                                                        \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
#define CUT_DEVICE_INIT(ARGC, ARGV)                                            \
  {                                                                            \
    int deviceCount;                                                           \
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                  \
    if (deviceCount == 0) {                                                    \
      fprintf(stderr, "cutil error: no devices supporting CUDA.\n");           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    int dev = 0;                                                               \
    cutGetCmdLineArgumenti(ARGC, (const char **)ARGV, "device", &dev);         \
    if (dev < 0)                                                               \
      dev = 0;                                                                 \
    if (dev > deviceCount - 1)                                                 \
      dev = deviceCount - 1;                                                   \
    cudaDeviceProp deviceProp;                                                 \
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));         \
    if (cutCheckCmdLineFlag(ARGC, (const char **)ARGV, "quiet") == CUTFalse)   \
      fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);          \
    CUDA_SAFE_CALL(cudaSetDevice(dev));                                        \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
//! Check for CUDA context lost
#define CUDA_CHECK_CTX_LOST(errorMessage)                                      \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    err = CUT_DEVICE_SYNCHRONIZE();                                            \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
//! Check for CUDA context lost
#define CU_CHECK_CTX_LOST(errorMessage)                                        \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (CUDA_ERROR_INVALID_CONTEXT != err) {                                   \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    err = CUT_DEVICE_SYNCHRONIZE();                                            \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",        \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
/*-------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------*/
/*#  define CUT_DEVICE_INIT_DRV(cuDevice, ARGC, ARGV) {                        \
  cuDevice = 0;                                                            \
  int deviceCount = 0;                                                     \
  CUresult err = cuInit(0);                                                \
  if (CUDA_SUCCESS == err)                                                 \
  CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&deviceCount));                \
  if (deviceCount == 0) {                                                  \
  fprintf(stderr, "cutil error: no devices supporting CUDA\n");        \
  exit(EXIT_FAILURE);                                                  \
  }                                                                        \
  int dev = 0;                                                             \
  cutGetCmdLineArgumenti(ARGC, (const char **) ARGV, "device", &dev);      \
  if (dev < 0) dev = 0;                                                    \
  if (dev > deviceCount-1) dev = deviceCount - 1;                          \
  CU_SAFE_CALL_NO_SYNC(cuDeviceGet(&cuDevice, dev));                       \
  char name[100];                                                          \
  cuDeviceGetName(name, 100, cuDevice);                                    \
  if (cutCheckCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == CUTFalse) \
  fprintf(stderr, "Using device %d: %s\n", dev, name);                  \
  }*/
/*-------------------------------------------------------------------------------*/

/*#define CUT_EXIT(argc, argv)                                                 \
  if (!cutCheckCmdLineFlag(argc, (const char**)argv, "noprompt")) {        \
  printf("\nPress ENTER to exit...\n");                                \
  fflush( stdout);                                                     \
  fflush( stderr);                                                     \
  getchar();                                                           \
  }                                                                        \
  exit(EXIT_SUCCESS);*/
/*-------------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif // #ifdef _DEBUG (else branch)
}

#endif
