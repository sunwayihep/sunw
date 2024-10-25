
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#ifdef MULTI_GPU
#include <mpi.h>
#endif

#include <string.h>
#include <iostream>
#include <iomanip>

#include <comm_mpi.h>
#include <complex.h>
#include <constants.h>
#include <alloc.h>
#include <cuda_common.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <exchange.h>

#include <tune.h>
#include <timer.h>
#include <texture_host.h>

using namespace std;

namespace CULQCD {

// Used some MILC functions

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

static int master_node = 0;
static int rank = -1;
static int size = -1;
static int gpuid = -1;
static int nx, ny, nz, nt;

static int squaresize[4];                /* dimensions of hypercubes */
static int nsquares[4] = { 1, 1, 1, 1 }; /* number of hypercubes in each
                                            direction */
static int machine_coordinates[4] = { 0, 0, 0, 0 }; /* logical machine
                                                       coordinates */

static int nodes_per_ionode[4];    /* dimensions of ionode partition */
static int *ionodegeomvals = NULL; /* ionode partitions */

int prime[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 };
#define MAXPRIMES (sizeof(prime) / sizeof(int))

static cudaDeviceProp deviceProp;
;

void SetMPIParam_MILC(const int latticedim[4], const int logical_coordinate[4],
                      const int nodesperdim[4]) {
  for (int i = 0; i < 4; i++) {
    machine_coordinates[i] = logical_coordinate[i];
    nsquares[i] = nodesperdim[i];
    squaresize[i] = latticedim[i] / nodesperdim[i];
  }
  nx = latticedim[0];
  ny = latticedim[1];
  nz = latticedim[2];
  nt = latticedim[3];
#ifdef MULTI_GPU
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
  rank = 0;
  size = 1;
#endif
  CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
}

static Timer culqcd_overalltime;

static TuneMode kerneltune = TUNE_NO;
static Verbosity verbose = SILENT;

TuneMode getTuning() { return kerneltune; }
Verbosity getVerbosity() { return verbose; }

void setTuning(TuneMode kerneltunein) { kerneltune = kerneltunein; }
void setVerbosity(Verbosity verbosein) { verbose = verbosein; }

void initCULQCD(int gpuidin, Verbosity verbosein, TuneMode tune) {
#ifdef MULTI_GPU
#ifdef MPI_GPU_DIRECT
  /* set CUDA-aware features environment variables enabled to "1" */
  setenv("MV2_USE_CUDA", "1", 1); // MVAPICH
  setenv("PMPI_GPU_AWARE", "1", 1); // IBM Platform MPI
  setenv("MPICH_RDMA_ENABLED_CUDA", "1", 1); // Cray
// Open MPI this feature are enabled per default

/*char* pPath;
pPath = getenv ("MV2_USE_CUDA");
if (pPath!=NULL)
  printf ("The current path is: %s",pPath); */
#endif

#ifdef MPI_MVAPICH
  int deviceCount = 1;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
  int nodeid = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
  CUDA_SAFE_CALL(cudaSetDevice(nodeid % deviceCount));
  CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
  cout << "Node id " << nodeid << " using CUDA device id " << gpuid << " with "
       << deviceCount << " total gpus." << endl;
#elif MPI_OPENMPI
  int deviceCount = 1;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
  int nodeid = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  CUDA_SAFE_CALL(cudaSetDevice(nodeid % deviceCount));
  CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
  cout << "Node id " << nodeid << " using CUDA device id " << gpuid << " with "
       << deviceCount << " total gpus." << endl;
#endif
  if (MPI_Init(NULL, NULL) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed.\n");
    exit(0);
  }
  culqcd_overalltime.start(); // uses MPI_Wtime and barrier to get time...
                              // cannot use before MPI_Init!!!!
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#if !defined(MPI_MVAPICH) && !defined(MPI_OPENMPI)
  int deviceCount = 1;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
  CUDA_SAFE_CALL(cudaSetDevice(rank % deviceCount));
  CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
  cout << "Node id " << rank << " using CUDA device id " << gpuid << " with "
       << deviceCount << " total gpus." << endl;
#endif
  printfCULQCD("Setup for %d nodes\n", size);
  MPI_Create_OP_DATATYPES();
#else
  culqcd_overalltime.start();
  gpuid = gpuidin;
  CUDA_SAFE_CALL(cudaSetDevice(gpuid));
  COUT << "Using CUDA device id " << gpuid << endl;
  rank = 0;
  size = 1;
#endif

  cudaGetDeviceProperties(&deviceProp, gpuid);

// IMPORTANT
// This feature must be disabled if MPICH_RDMA_ENABLED_CUDA is set to one and
// using GPU comms...
// Probably a bug in cray compiler...
// if(deviceProp.canMapHostMemory) cudaSetDeviceFlags(cudaDeviceMapHost);

#ifdef GLOBAL_SET_CACHE_PREFER_L1
  //---------------------------------------------------------------------------------------
  // Set preferred cache configuration, options:
  //  - cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
  //  - cudaFuncCachePreferShared: prefer larger shared memory and smaller L1
  // cache
  //  - cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
  //---------------------------------------------------------------------------------------
  CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
#warning Set cudaDeviceSetCacheConfig() to cudaFuncCachePreferL1
#endif
  master_node = 0;
  setVerbosity(verbosein);
  setTuning(tune);
  if (tune == TUNE_YES)
    loadTuneCache(verbosein);
}

void EndCULQCD(int status) {
  if (getTuning() == TUNE_YES)
    saveTuneCache(getVerbosity());
  UNBIND_ALL_TEXTURES();
#ifdef MULTI_GPU
  FreeTempBuffersAndStreams();
#endif
  // Prints peak memory used by host and device
  printPeakMemUsage();
  // Prints all memory alocatted using alloc.h functions but not freed
  assertAllMemFree();
  // Free all memory still allocated
  if (!status)
    FreeAllMemory();
#ifndef MPI_GPU_DIRECT
  CUT_DEVICE_RESET();
#endif
  culqcd_overalltime.stop();
#ifdef MULTI_GPU
  MPI_Release_OP_DATATYPES();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  COUT << "Total time: " << culqcd_overalltime.getElapsedTimeInSec() << " s"
       << std::endl;
  printfCULQCD("Exiting...\n");
  exit(status);
}

/**  broadcast from rank 0 */
void comm_broadcast(void *data, size_t nbytes) {
#ifdef MULTI_GPU
  MPI_CHECK(MPI_Bcast(data, (int)nbytes, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif
}

int nodes_per_dim(int dim) { return nsquares[dim]; }
void logical_coordinate(int coords[]) {
  for (int d = 0; d < 4; d++)
    coords[d] = machine_coordinates[d];
}

#ifdef MULTI_GPU
void comm_abort(int status) {
  MPI_CHECK(MPI_Finalize());
  exit(status);
}
#else
void comm_abort(int status) { exit(status); }
#endif

/*
**  Return my node number
*/
int mynode(void) { return rank; }

int masternode() { return master_node; }

/*
**  Return number of nodes
*/
int numnodes(void) { return size; }

bool comm_dim_partitioned(int dim) {
  if (nsquares[dim] > 1)
    return true;
  return false;
}

int comm_rank(void) { return rank; }

int comm_size(void) { return size; }

int comm_gpuid(void) { return gpuid; }

int node_number(int x, int y, int z, int t) {
  register int i;
  x /= squaresize[XUP];
  y /= squaresize[YUP];
  z /= squaresize[ZUP];
  t /= squaresize[TUP];
  i = x + nsquares[XUP] * (y + nsquares[YUP] * (z + nsquares[ZUP] * (t)));
  return (i);
}

/*------------------------------------------------------------------*/
/* Convert rank to coordinates */
void lex_coords(int coords[], const int dim, const int size[],
                const size_t rank) {
  int d;
  size_t r = rank;

  for (d = 0; d < dim; d++) {
    coords[d] = r % size[d];
    r /= size[d];
  }
}

/*------------------------------------------------------------------*/
/* Parity of the coordinate */
static int coord_parity(int r[]) { return (r[0] + r[1] + r[2] + r[3]) % 2; }

/*------------------------------------------------------------------*/
/* Convert coordinate to linear lexicographic rank (inverse of
   lex_coords) */

static size_t lex_rank(const int coords[], int dim, int size[]) {
  int d;
  size_t rank = coords[dim - 1];

  for (d = dim - 2; d >= 0; d--) {
    rank = rank * size[d] + coords[d];
  }
  return rank;
}

/*--------------------------------------------------------------------*/
void setup_hyper_prime(int _nx, int _ny, int _nz, int _nt) {

  int i, j, k, dir;
  nx = _nx;
  ny = _ny;
  nz = _nz;
  nt = _nt;
  /*if(mynode()==0){
    printf("hyper_prime,");
    printf("\n");
  }*/

  /* Figure out dimensions of rectangle */
  squaresize[XUP] = nx;
  squaresize[YUP] = ny;
  squaresize[ZUP] = nz;
  squaresize[TUP] = nt;
  nsquares[XUP] = nsquares[YUP] = nsquares[ZUP] = nsquares[TUP] = 1;
// if(0){
#ifdef START_LATTICE_PARTITION_BY_X
  i = 1; /* current number of hypercubes */
  while (i < numnodes()) {
    /* figure out which prime to divide by starting with largest */
    k = MAXPRIMES - 1;
    while ((numnodes() / i) % prime[k] != 0 && k > 0)
      --k;
    /* figure out which direction to divide */

    /* find largest dimension of h-cubes divisible by prime[k] */
    for (j = 0, dir = XUP; dir <= TUP; dir++)
      if (squaresize[dir] > j && squaresize[dir] % prime[k] == 0)
        j = squaresize[dir];

    /* if one direction with largest dimension has already been
       divided, divide it again.  Otherwise divide first direction
       with largest dimension. */
    for (dir = XUP; dir <= TUP; dir++)
      if (squaresize[dir] == j && nsquares[dir] > 1)
        break;
    if (dir > TUP)
      for (dir = XUP; dir <= TUP; dir++)
        if (squaresize[dir] == j)
          break;
    /* This can fail if I run out of prime factors in the dimensions */
    if (dir > TUP) {
      if (mynode() == 0)
        printf("LAYOUT: Can't lay out this lattice, not enough factors of %d\n",
               prime[k]);
      exit(1);
    }

    /* do the surgery */
    i *= prime[k];
    squaresize[dir] /= prime[k];
    nsquares[dir] *= prime[k];
  }
#else
  /*}
  else{*/

  i = 1; /* current number of hypercubes */
  while (i < numnodes()) {
    /* figure out which prime to divide by starting with largest */
    k = MAXPRIMES - 1;
    while ((numnodes() / i) % prime[k] != 0 && k > 0)
      --k;
    /* figure out which direction to divide */

    /* find largest dimension of h-cubes divisible by prime[k] */
    // for(j=0,dir=XUP;dir<=TUP;dir++)
    for (j = 0, dir = TUP; dir >= XUP; dir--)
      if (squaresize[dir] > j && squaresize[dir] % prime[k] == 0)
        j = squaresize[dir];

    /* if one direction with largest dimension has already been
       divided, divide it again.  Otherwise divide first direction
       with largest dimension. */
    // for(dir=XUP;dir<=TUP;dir++)
    for (dir = TUP; dir >= XUP; dir--)
      if (squaresize[dir] == j && nsquares[dir] > 1)
        break;
    // if( dir > TUP)
    // for(dir=XUP;dir<=TUP;dir++)
    if (dir < XUP)
      for (dir = TUP; dir >= XUP; dir--)
        if (squaresize[dir] == j)
          break;
    /* This can fail if I run out of prime factors in the dimensions */
    if (dir < XUP) {
      // if( dir > TUP){
      if (mynode() == 0)
        printf("LAYOUT: Can't lay out this lattice, not enough factors of %d\n",
               prime[k]);
      exit(1);
    }

    /* do the surgery */
    i *= prime[k];
    squaresize[dir] /= prime[k];
    nsquares[dir] *= prime[k];
  }
//}
#endif

  lex_coords(machine_coordinates, 4, nsquares, mynode());
  // printf("#%d$%d::::%d:%d:%d:%d@@@@%d:%d:%d:%d\n",numnodes(),mynode(),
  // squaresize[0], squaresize[1], squaresize[2],
  // squaresize[3],nsquares[0],nsquares[1],nsquares[2],nsquares[3]);
}

#if defined(MULTI_GPU)

void comm_allreduce_array(double *data, size_t size) {
  double *recvbuf = new double[size];
  MPI_CHECK(
      MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  memcpy(data, recvbuf, size * sizeof(double));
  delete[] recvbuf;
}

void comm_allreduce_array(float *data, size_t size) {
  float *recvbuf = new float[size];
  MPI_CHECK(
      MPI_Allreduce(data, recvbuf, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
  memcpy(data, recvbuf, size * sizeof(float));
  delete[] recvbuf;
}

void comm_allreduce(float *data) {
  float recvbuf;
  MPI_CHECK(
      MPI_Allreduce(data, &recvbuf, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
  *data = recvbuf;
}

void comm_allreduce(double *data) {
  double recvbuf;
  MPI_CHECK(
      MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  *data = recvbuf;
}

void comm_allreduce_max(double *data) {
  double recvbuf;
  MPI_CHECK(
      MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  *data = recvbuf;
}
void comm_allreduce_max(float *data) {
  float recvbuf;
  MPI_CHECK(
      MPI_Allreduce(data, &recvbuf, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));
  *data = recvbuf;
}
void comm_allreduce_array_max(double *data, size_t size) {
  double *recvbuf = new double[size];
  MPI_CHECK(
      MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  memcpy(data, recvbuf, size * sizeof(double));
  delete[] recvbuf;
}

void comm_allreduce_array_max(float *data, size_t size) {
  float *recvbuf = new float[size];
  MPI_CHECK(
      MPI_Allreduce(data, recvbuf, size, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));
  memcpy(data, recvbuf, size * sizeof(float));
  delete[] recvbuf;
}

void comm_allreduce_min(double *data) {
  double recvbuf;
  MPI_CHECK(
      MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
  *data = recvbuf;
}

void comm_allreduce_min(float *data) {
  float recvbuf;
  MPI_CHECK(
      MPI_Allreduce(data, &recvbuf, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD));
  *data = recvbuf;
}

void comm_allreduce_array_min(double *data, size_t size) {
  double *recvbuf = new double[size];
  MPI_CHECK(
      MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
  memcpy(data, recvbuf, size * sizeof(double));
  delete[] recvbuf;
}

void comm_allreduce_array_min(float *data, size_t size) {
  float *recvbuf = new float[size];
  MPI_CHECK(
      MPI_Allreduce(data, recvbuf, size, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD));
  memcpy(data, recvbuf, size * sizeof(float));
  delete[] recvbuf;
}

void SingleComplexMax(complexs *in, complexs *inout, int *len,
                      MPI_Datatype *cType) {
  int i;
  complexs c;
  for (i = 0; i < *len; ++i) {
    if (in->abs() > inout->abs())
      c = (*in);
    else
      c = (*inout);
    *inout = c;
    in++;
    inout++;
  }
}
void DoubleComplexMax(complexd *in, complexd *inout, int *len,
                      MPI_Datatype *cType) {
  int i;
  complexd c;
  for (i = 0; i < *len; ++i) {
    if (in->abs() > inout->abs())
      c = (*in);
    else
      c = (*inout);
    *inout = c;
    in++;
    inout++;
  }
}

void SingleComplexSum(complexs *in, complexs *inout, int *len,
                      MPI_Datatype *cType) {
  int i;
  complexs c;
  for (i = 0; i < *len; ++i) {
    // c = (*inout)+(*in);
    c.val.x = in->val.x + inout->val.x;
    c.val.y = inout->val.y + in->val.y;
    *inout = c;
    in++;
    inout++;
  }
}
void DoubleComplexSum(complexd *in, complexd *inout, int *len,
                      MPI_Datatype *cType) {
  int i;
  complexd c;
  for (i = 0; i < *len; ++i) {
    c.val.x = in->val.x + inout->val.x;
    c.val.y = inout->val.y + in->val.y;
    // For some strange reason in one machine this only works if in and inout
    // are crossed...
    *inout = c;
    in++;
    inout++;
  }
}

MPI_Op MPI_SINGLE_COMPLEX_MAX;
MPI_Op MPI_DOUBLE_COMPLEX_MAX;

// MPI complex single precision sum operation
MPI_Op MPI_SINGLE_COMPLEX_SUM;
// MPI complex single precision datatype
MPI_Datatype ComplexSingle;
// MPI complex double precision sum operation
MPI_Op MPI_DOUBLE_COMPLEX_SUM;
// MPI complex double precision datatype
MPI_Datatype ComplexDouble;
MPI_Datatype SingleMSUN;
MPI_Datatype DoubleMSUN;
#if (NCOLORS == 3)
MPI_Datatype SingleMSUN12;
MPI_Datatype DoubleMSUN12;
MPI_Datatype SingleMSUN8;
MPI_Datatype DoubleMSUN8;
#endif
MPI_Status MPI_StatuS;

template <class T> void comm_Allreduce(T *data) {
  T recvbuf;
  MPI_Allreduce(data, &recvbuf, 1, mpi_datatype<T>(), mpi_sumtype<T>(),
                MPI_COMM_WORLD);
  *data = recvbuf;
}

template void comm_Allreduce<float>(float *data);
template void comm_Allreduce<double>(double *data);
template void comm_Allreduce<complexs>(complexs *data);
template void comm_Allreduce<complexd>(complexd *data);

template <class T> void comm_Allreduce(T *data, size_t size) {
  T *recvbuf = new T[size];
  MPI_Allreduce(data, recvbuf, size, mpi_datatype<T>(), mpi_sumtype<T>(),
                MPI_COMM_WORLD);
  memcpy(data, recvbuf, size * sizeof(double));
  delete[] recvbuf;
}

template void comm_Allreduce<float>(float *data, size_t size);
template void comm_Allreduce<double>(double *data, size_t size);
template void comm_Allreduce<complexs>(complexs *data, size_t size);
template void comm_Allreduce<complexd>(complexd *data, size_t size);

template <class T> void comm_Allreduce_Max(T *data) {
  T recvbuf;
  MPI_Allreduce(data, &recvbuf, 1, mpi_datatype<T>(), mpi_maxtype<T>(),
                MPI_COMM_WORLD);
  *data = recvbuf;
}

template void comm_Allreduce_Max<float>(float *data);
template void comm_Allreduce_Max<double>(double *data);
template void comm_Allreduce_Max<complexs>(complexs *data);
template void comm_Allreduce_Max<complexd>(complexd *data);
#endif

void MPI_Create_OP_DATATYPES() {
#if defined(MULTI_GPU)
  // Set mpi complex sum reduction operation
  MPI_Type_contiguous(2, MPI_FLOAT, &ComplexSingle);
  MPI_Type_commit(&ComplexSingle);
  MPI_Op_create((MPI_User_function *)SingleComplexSum, true,
                &MPI_SINGLE_COMPLEX_SUM);
  MPI_Op_create((MPI_User_function *)SingleComplexMax, true,
                &MPI_SINGLE_COMPLEX_MAX);

  MPI_Type_contiguous(2, MPI_DOUBLE, &ComplexDouble);
  MPI_Type_commit(&ComplexDouble);
  MPI_Op_create((MPI_User_function *)DoubleComplexSum, true,
                &MPI_DOUBLE_COMPLEX_SUM);
  MPI_Op_create((MPI_User_function *)DoubleComplexMax, true,
                &MPI_DOUBLE_COMPLEX_MAX);

  MPI_Type_contiguous(NCOLORS * NCOLORS, MPI_FLOAT, &SingleMSUN);
  MPI_Type_commit(&SingleMSUN);
  MPI_Type_contiguous(NCOLORS * NCOLORS, MPI_DOUBLE, &DoubleMSUN);
  MPI_Type_commit(&DoubleMSUN);
#if (NCOLORS == 3)
  MPI_Type_contiguous(12, MPI_FLOAT, &SingleMSUN12);
  MPI_Type_commit(&SingleMSUN12);
  MPI_Type_contiguous(12, MPI_DOUBLE, &DoubleMSUN12);
  MPI_Type_commit(&DoubleMSUN12);
  MPI_Type_contiguous(8, MPI_FLOAT, &SingleMSUN8);
  MPI_Type_commit(&SingleMSUN8);
  MPI_Type_contiguous(8, MPI_DOUBLE, &DoubleMSUN8);
  MPI_Type_commit(&DoubleMSUN8);
#endif

#endif
}

void MPI_Release_OP_DATATYPES() {
#if defined(MULTI_GPU)
  MPI_Op_free(&MPI_SINGLE_COMPLEX_MAX);
  MPI_Op_free(&MPI_DOUBLE_COMPLEX_MAX);
  MPI_Op_free(&MPI_SINGLE_COMPLEX_SUM);
  MPI_Op_free(&MPI_DOUBLE_COMPLEX_SUM);
  MPI_Type_free(&ComplexSingle);
  MPI_Type_free(&ComplexDouble);
  MPI_Type_free(&SingleMSUN);
  MPI_Type_free(&DoubleMSUN);
#if (NCOLORS == 3)
  MPI_Type_free(&SingleMSUN12);
  MPI_Type_free(&DoubleMSUN12);
  MPI_Type_free(&SingleMSUN8);
  MPI_Type_free(&DoubleMSUN8);
#endif
#endif
}
}
