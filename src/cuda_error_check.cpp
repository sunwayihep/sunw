#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <map>
#include <unistd.h>   // for getpagesize()
#include <execinfo.h> // for backtrace
#include <iostream>

#include <cuda_common.h>
#include <modes.h>
#include <comm_mpi.h>

#include "timer.h"
#include "cuda_error_check.h"
//#include <climits>

namespace CULQCD {

void cudaDevSync_(const char *func, const char *file, int line) {
  cudaSafeCall_(func, file, line, cudaDeviceSynchronize());
}

void cudaSafeCall_(const char *func, const char *file, int line,
                   cudaError_t call) {
  cudaError_t err = call;
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA error in file %s:%i in %s: %s.\n", file, line, func,
            cudaGetErrorString(err));
    errorCULQCD("Aborting");
  }
}

void cudaCheckError_(const char *func, const char *file, int line,
                     const char *errorMessage) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s in file %s:%i in %s: %s.\n", errorMessage,
            file, line, func, cudaGetErrorString(err));
    errorCULQCD("Aborting");
  }
#ifdef _DEBUG
  cudaDevSync();
#endif
}

static const char *_cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

void cublasSafeCall_(const char *func, const char *file, int line,
                     cublasStatus_t call) {
  cublasStatus_t err = call;
  if (CUBLAS_STATUS_SUCCESS != err) {
    fprintf(stderr, "CUBLAS error in file %s:%i in %s: %s.\n", file, line, func,
            _cublasGetErrorEnum(err));
    errorCULQCD("Aborting");
  }
}

static const char *_cusparseGetErrorEnum(cusparseStatus_t error) {
  switch (error) {
  case CUSPARSE_STATUS_SUCCESS:
    return "CUSPARSE_STATUS_SUCCESS";
  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";
  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "CUSPARSE_STATUS_ALLOC_FAILED";
  case CUSPARSE_STATUS_INVALID_VALUE:
    return "CUSPARSE_STATUS_INVALID_VALUE";
  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "CUSPARSE_STATUS_ARCH_MISMATCH";
  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "CUSPARSE_STATUS_MAPPING_ERROR";
  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";
  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "CUSPARSE_STATUS_INTERNAL_ERROR";
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSPARSE_STATUS_ZERO_PIVOT:
    return "CUSPARSE_STATUS_ZERO_PIVOT";
  }
  return "<unknown>";
}
void cusparseSafeCall_(const char *func, const char *file, int line,
                       cusparseStatus_t call) {
  cusparseStatus_t err = call;
  if (CUSPARSE_STATUS_SUCCESS != err) {
    fprintf(stderr, "CUSPARSE error in file %s:%i in %s: %s.\n", file, line,
            func, _cusparseGetErrorEnum(err));
    errorCULQCD("Aborting");
  }
}

static const char *_cusolverGetErrorEnum(cusolverStatus_t error) {
  switch (error) {
  case CUSOLVER_STATUS_SUCCESS:
    return "CUSOLVER_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_MAPPING_ERROR:
    return "CUSOLVER_STATUS_MAPPING_ERROR";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_NOT_SUPPORTED";
  case CUSOLVER_STATUS_ZERO_PIVOT:
    return "CUSOLVER_STATUS_ZERO_PIVOT";
  case CUSOLVER_STATUS_INVALID_LICENSE:
    return "CUSOLVER_STATUS_INVALID_LICENSE";
  }
  return "<unknown>";
}
void cusolverSafeCall_(const char *func, const char *file, int line,
                       cusolverStatus_t call) {
  cusolverStatus_t err = call;
  if (CUSOLVER_STATUS_SUCCESS != err) {
    fprintf(stderr, "CUSOLVER error in file %s:%i in %s: %s.\n", file, line,
            func, _cusolverGetErrorEnum(err));
    errorCULQCD("Aborting");
  }
}

static const char *_curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

void curandSafeCall_(const char *func, const char *file, int line,
                     curandStatus_t call) {
  curandStatus_t err = call;
  if (CURAND_STATUS_SUCCESS != err) {
    fprintf(stderr, "CUSOLVER error in file %s:%i in %s: %s.\n", file, line,
            func, _curandGetErrorString(err));
    errorCULQCD("Aborting");
  }
}
}
