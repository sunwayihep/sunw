
#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cstdlib>
//#include <enum_sbreak.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <curand.h>

#include "alloc.h"

namespace CULQCD {

void cudaDevSync_(const char *func, const char *file, int line);
void cudaSafeCall_(const char *func, const char *file, int line,
                   cudaError_t call);
void cudaCheckError_(const char *func, const char *file, int line,
                     const char *errorMessage);
void cusparseSafeCall_(const char *func, const char *file, int line,
                       cusparseStatus_t call);
void cusolverSafeCall_(const char *func, const char *file, int line,
                       cusolverStatus_t call);
void cublasSafeCall_(const char *func, const char *file, int line,
                     cublasStatus_t call);
void curandSafeCall_(const char *func, const char *file, int line,
                     curandStatus_t call);
}

#define cudaDeviceSync()                                                       \
  CULQCD::cudaDevSync_(__func__, CULQCD::file_name(__FILE__), __LINE__)
#define cudaSafeCall(call)                                                     \
  CULQCD::cudaSafeCall_(__func__, CULQCD::file_name(__FILE__), __LINE__, call)
#define cusparseSafeCall(call)                                                 \
  CULQCD::cusparseSafeCall_(__func__, CULQCD::file_name(__FILE__), __LINE__,   \
                            call)
#define cusolverSafeCall(call)                                                 \
  CULQCD::cusolverSafeCall_(__func__, CULQCD::file_name(__FILE__), __LINE__,   \
                            call)
#define cublasSafeCall(call)                                                   \
  CULQCD::cublasSafeCall_(__func__, CULQCD::file_name(__FILE__), __LINE__, call)
#define curandSafeCall(call)                                                   \
  CULQCD::curandSafeCall_(__func__, CULQCD::file_name(__FILE__), __LINE__, call)
#define cudaCheckError(errorMessage)                                           \
  CULQCD::cudaCheckError_(__func__, CULQCD::file_name(__FILE__), __LINE__,     \
                          errorMessage)

#endif
