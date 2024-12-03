

#ifndef _COMM_MPI_H
#define _COMM_MPI_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef MULTI_GPU
#include <mpi.h>
#endif

#include <iomanip>
#include <iostream>
#include <string.h>

#include <complex.h>
#include <constants.h>
#include <matrixsun.h>
#include <modes.h>

namespace CULQCD {

TuneMode getTuning();
Verbosity getVerbosity();

void setTuning(TuneMode kerneltunein);
void setVerbosity(Verbosity verbosein);

void initCULQCD(int gpuidin, Verbosity verbosein = SILENT,
                TuneMode tune = TUNE_NO);

void EndCULQCD(int status);

void SetMPIParam_MILC(const int latticedim[NDIMS],
                      const int logical_coordinate[NDIMS],
                      const int nodesperdim[NDIMS]);

#ifdef MULTI_GPU
void comm_abort(int status);
#else
void comm_abort(int status);
#endif

int GetDeviceId();

int comm_rank(void);

int comm_size(void);

int comm_gpuid(void);
int node_number(int x, int y, int z, int t);

int mynode(void);
int masternode();
int numnodes(void);
bool comm_dim_partitioned(int dim);

void comm_broadcast(void *data, size_t nbytes);

void logical_coordinate(int coords[NDIMS]);
int nodes_per_dim(int dim);
void setup_hyper_prime(int _nx, int _ny, int _nz, int _nt);

#ifdef MULTI_GPU
#define printfCULQCD(abc, ...)                                                 \
  if (mynode() == masternode())                                                \
    printf(abc, ##__VA_ARGS__);                                                \
  else                                                                         \
    (void)0
#define COUT                                                                   \
  if (mynode() == masternode())                                                \
  std::cout
//    else (void)0

// MPI error handling macro
#define MPI_CHECK(call)                                                        \
  if ((call) != MPI_SUCCESS) {                                                 \
    cerr << "MPI error calling \"" #call "\"\n";                               \
    MPI_Abort(MPI_COMM_WORLD, (-1));                                           \
  } else                                                                       \
    (void)0
#define printfError(abc, ...)                                                  \
  do {                                                                         \
    printf("Error in %d, " __FILE__ ": %d in %s()\n\t", mynode(), __LINE__,    \
           __func__);                                                          \
    printf(abc, ##__VA_ARGS__);                                                \
    MPI_Abort(MPI_COMM_WORLD, (-1));                                           \
    EndCULQCD(1);                                                              \
  } while (0)
#define errorCULQCD(abc, ...)                                                  \
  do {                                                                         \
    printf("Error in %d, " __FILE__ ": %d in %s()\n\t", mynode(), __LINE__,    \
           __func__);                                                          \
    printf(abc, ##__VA_ARGS__);                                                \
    MPI_Abort(MPI_COMM_WORLD, (-1));                                           \
    EndCULQCD(1);                                                              \
  } while (0)
#else
#define printfCULQCD(abc, ...) printf(abc, ##__VA_ARGS__);
#define COUT std::cout
#define printfError(abc, ...)                                                  \
  do {                                                                         \
    printf("Error in " __FILE__ ": %d in %s()\n\t", __LINE__, __func__);       \
    printf(abc, ##__VA_ARGS__);                                                \
    EndCULQCD(1);                                                              \
  } while (0)
#define errorCULQCD(abc, ...)                                                  \
  do {                                                                         \
    printf("Error in " __FILE__ ": %d in %s()\n\t", __LINE__, __func__);       \
    printf(abc, ##__VA_ARGS__);                                                \
    EndCULQCD(1);                                                              \
  } while (0)
#endif

#define checkCudaErrorNoSync()                                                 \
  do {                                                                         \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess)                                                  \
      errorCULQCD("(CUDA) %s", cudaGetErrorString(error));                     \
  } while (0)

#ifdef HOST_DEBUG
#define checkCudaError()                                                       \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    checkCudaErrorNoSync();                                                    \
  } while (0)

#else

#define checkCudaError() checkCudaErrorNoSync()
#endif

#ifdef MULTI_GPU

void SingleComplexSum(complexs *in, complexs *inout, int *len,
                      MPI_Datatype *cType);
void DoubleComplexSum(complexd *in, complexd *inout, int *len,
                      MPI_Datatype *cType);
void SingleComplexMax(complexs *in, complexs *inout, int *len,
                      MPI_Datatype *cType);
void DoubleComplexMax(complexd *in, complexd *inout, int *len,
                      MPI_Datatype *cType);

template <class Real>
void MSUNProduct(msun *inVec, msun *inOutVec, int *length, MPI_Datatype *cType);

extern MPI_Op MPI_SINGLE_COMPLEX_SUM;
extern MPI_Datatype ComplexSingle;
extern MPI_Op MPI_DOUBLE_COMPLEX_SUM;
extern MPI_Datatype ComplexDouble;

extern MPI_Datatype SingleMSUN;
extern MPI_Datatype DoubleMSUN;
#if (NCOLORS == 3)
extern MPI_Datatype SingleMSUN12;
extern MPI_Datatype DoubleMSUN12;
extern MPI_Datatype SingleMSUN8;
extern MPI_Datatype DoubleMSUN8;
#endif

extern MPI_Op MPI_SINGLE_COMPLEX_MAX;
extern MPI_Op MPI_DOUBLE_COMPLEX_MAX;

extern MPI_Status MPI_StatuS;

/**
    @brief templated data types for MPI reductions
*/
template <class T> inline MPI_Datatype mpi_datatype() {
  return MPI_DATATYPE_NULL;
}
template <> inline MPI_Datatype mpi_datatype<int>() { return MPI_INT; }
template <> inline MPI_Datatype mpi_datatype<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_datatype<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_datatype<complexs>() {
  return ComplexSingle;
}
template <> inline MPI_Datatype mpi_datatype<complexd>() {
  return ComplexDouble;
}

template <class T> inline MPI_Op mpi_sumtype() { return MPI_OP_NULL; }
template <> inline MPI_Op mpi_sumtype<complexs>() {
  return MPI_SINGLE_COMPLEX_SUM;
}
template <> inline MPI_Op mpi_sumtype<complexd>() {
  return MPI_DOUBLE_COMPLEX_SUM;
}
template <> inline MPI_Op mpi_sumtype<float>() { return MPI_SUM; }
template <> inline MPI_Op mpi_sumtype<double>() { return MPI_SUM; }

template <class T> inline MPI_Op mpi_maxtype() { return MPI_OP_NULL; }
template <> inline MPI_Op mpi_maxtype<complexs>() {
  return MPI_SINGLE_COMPLEX_MAX;
}
template <> inline MPI_Op mpi_maxtype<complexd>() {
  return MPI_DOUBLE_COMPLEX_MAX;
}
template <> inline MPI_Op mpi_maxtype<float>() { return MPI_MAX; }
template <> inline MPI_Op mpi_maxtype<double>() { return MPI_MAX; }

template <class T> void comm_Allreduce_Max(T *data);

template <class T> void comm_Allreduce(T *data);
template <class T> void comm_Allreduce(T *data, size_t size);

void comm_allreduce(float *data);
void comm_allreduce(double *data);
void comm_allreduce_array(double *data, size_t size);
void comm_allreduce_array(float *data, size_t size);
void comm_allreduce_max(double *data);
void comm_allreduce_max(float *data);
void comm_allreduce_min(double *data);
void comm_allreduce_min(float *data);
void comm_allreduce_array_max(double *data, size_t size);
void comm_allreduce_array_max(float *data, size_t size);
void comm_allreduce_array_min(double *data, size_t size);
void comm_allreduce_array_min(float *data, size_t size);

#endif

void MPI_Create_OP_DATATYPES();
void MPI_Release_OP_DATATYPES();
} // namespace CULQCD

#endif //_COMM_MPI_H
