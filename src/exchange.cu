
#ifdef MULTI_GPU
#include <mpi.h>
#endif

#include <comm_mpi.h>
#include <complex.h>
#include <constants.h>
#include <device_load_save.h>
#include <gaugearray.h>
#include <index.h>
#include <matrixsun.h>
#include <timer.h>

#include <cuda.h>
#include <thrust/binary_search.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <exchange.h>
#include <texture_host.h>

namespace CULQCD {
#ifdef MULTI_GPU
// temporary buffers for data exchange between node neighbors
static void *sendbuff_gpu = NULL;
static void *recvbuff_gpu = NULL;
#ifndef MPI_GPU_DIRECT
static void *sendbuff_cpu = NULL;
static void *recvbuff_cpu = NULL;
#endif

static void *sendbuff = NULL;
static void *recvbuff = NULL;
// offset for each active face links used in send receive buffers
static size_t offsetptr[4];

// CUDA streams
static int numberStreams;
static cudaStream_t *exchangeStream;

/*static MPI_Request send_request_border[4];
static MPI_Request recv_request_border[4];*/

// 3 for SOA, SOA12 and SOA8
// 4 for lattice dimensions
static MPI_Request top_send_request_border[3][4];
static MPI_Request top_recv_request_border[3][4];
static MPI_Request bot_send_request_border[3][4];
static MPI_Request bot_recv_request_border[3][4];

static bool allocated = false;

int GetIdFromNumElems(int NumElems) {
#if (NCOLORS == 3)
  if (NumElems == 9)
    return 0;
  else if (NumElems == 6)
    return 1;
  else
    return 2;
#else
  return 0;
#endif
}

template <class Real> void AllocateTempBuffersAndStreams() {
  if (numnodes() == 1)
    return;
  if (allocated)
    return;

  numberStreams = 8;
  exchangeStream = (cudaStream_t *)malloc(numberStreams * sizeof(cudaStream_t));
  for (int i = 0; i < numberStreams; i++)
    CUDA_SAFE_CALL(cudaStreamCreate(&exchangeStream[i]));
  int ncopyelems = 0;
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    ncopyelems += PARAMS::FaceSizeG[PARAMS::FaceId[fc]];
    offsetptr[fc] = 0;
    for (int i = 0; i < fc; i++)
      offsetptr[fc] += PARAMS::FaceSizeG[PARAMS::FaceId[i]] * NCOLORS * NCOLORS;
  }
  ncopyelems *= NCOLORS * NCOLORS;
  sendbuff_gpu = dev_malloc(ncopyelems * sizeof(complex));
  recvbuff_gpu = dev_malloc(ncopyelems * sizeof(complex));

#ifdef MPI_GPU_DIRECT
  sendbuff = sendbuff_gpu;
  recvbuff = recvbuff_gpu;
#else
  sendbuff_cpu = pinned_malloc(ncopyelems * sizeof(complex));
  recvbuff_cpu = pinned_malloc(ncopyelems * sizeof(complex));
  sendbuff = sendbuff_cpu;
  recvbuff = recvbuff_cpu;
#endif

// set init transfers to all array types....
#if (NCOLORS == 3)
  int maxl = 3;
#else
  int maxl = 1;
#endif
  for (int tpe = 0; tpe < maxl; tpe++) {
    int NumElems = NCOLORS * NCOLORS;
    if (tpe == 1)
      NumElems = 6;
    else if (tpe == 2)
      NumElems = 4;
    for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {

      int faceid = PARAMS::FaceId[fc];
      int facesize = PARAMS::FaceSizeG[faceid] >> 1;
      int ncopyelemshalf = facesize * NumElems;

      // top face
      MPI_CHECK(MPI_Send_init(
          reinterpret_cast<complex *>(sendbuff) + offsetptr[fc], ncopyelemshalf,
          mpi_datatype<complex>(), PARAMS::NodeIdRight[fc], fc, MPI_COMM_WORLD,
          &top_send_request_border[tpe][fc]));

      MPI_CHECK(MPI_Recv_init(
          reinterpret_cast<complex *>(recvbuff) + offsetptr[fc], ncopyelemshalf,
          mpi_datatype<complex>(), PARAMS::NodeIdLeft[fc], fc, MPI_COMM_WORLD,
          &top_recv_request_border[tpe][fc]));
      // bottom face
      MPI_CHECK(MPI_Send_init(reinterpret_cast<complex *>(sendbuff) +
                                  offsetptr[fc] + ncopyelemshalf,
                              ncopyelemshalf, mpi_datatype<complex>(),
                              PARAMS::NodeIdLeft[fc], fc, MPI_COMM_WORLD,
                              &bot_send_request_border[tpe][fc]));

      MPI_CHECK(MPI_Recv_init(reinterpret_cast<complex *>(recvbuff) +
                                  offsetptr[fc] + ncopyelemshalf,
                              ncopyelemshalf, mpi_datatype<complex>(),
                              PARAMS::NodeIdRight[fc], fc, MPI_COMM_WORLD,
                              &bot_recv_request_border[tpe][fc]));
    }
  }
  allocated = true;
}
template void AllocateTempBuffersAndStreams<float>();
template void AllocateTempBuffersAndStreams<double>();

void FreeTempBuffersAndStreams() {
  if (numnodes() == 1)
    return;
  if (allocated) {
    for (int i = 0; i < numberStreams; i++)
      CUDA_SAFE_CALL(cudaStreamDestroy(exchangeStream[i]));
    free(exchangeStream);
    dev_free(sendbuff_gpu);
    dev_free(recvbuff_gpu);
#ifndef MPI_GPU_DIRECT
    host_free(sendbuff_cpu);
    host_free(recvbuff_cpu);
#endif
    allocated = false;
  }
}

inline __device__ int LatticeFaceIndex00(int idd, int oddbit, int faceid,
                                         int borderid) {
  int idx, za, xodd, x[4];
  switch (faceid) {
  case 0: // X FACE
    za = idd / (DEVPARAMS::GridWGhost[1] / 2);
    x[3] = za / DEVPARAMS::GridWGhost[2];
    x[2] = za - x[3] * DEVPARAMS::GridWGhost[2];
    xodd = (borderid + x[2] + x[3] + oddbit) & 1;
    x[1] = (2 * idd + xodd) - za * DEVPARAMS::GridWGhost[1];
    x[0] = borderid;
    break;
  case 1: // Y FACE
    za = idd / (DEVPARAMS::GridWGhost[0] / 2);
    x[3] = za / DEVPARAMS::GridWGhost[2];
    x[2] = za - x[3] * DEVPARAMS::GridWGhost[2];
    xodd = (borderid + x[2] + x[3] + oddbit) & 1;
    x[0] = (2 * idd + xodd) - za * DEVPARAMS::GridWGhost[0];
    x[1] = borderid;
    break;
  case 2: // Z FACE
    za = idd / (DEVPARAMS::GridWGhost[0] / 2);
    x[3] = za / DEVPARAMS::GridWGhost[1];
    x[1] = za - x[3] * DEVPARAMS::GridWGhost[1];
    xodd = (borderid + x[1] + x[3] + oddbit) & 1;
    x[0] = (2 * idd + xodd) - za * DEVPARAMS::GridWGhost[0];
    x[2] = borderid;
    break;
  case 3: // T FACE
    za = idd / (DEVPARAMS::GridWGhost[0] / 2);
    x[2] = za / DEVPARAMS::GridWGhost[1];
    x[1] = za - x[2] * DEVPARAMS::GridWGhost[1];
    xodd = (borderid + x[1] + x[2] + oddbit) & 1;
    x[0] = (2 * idd + xodd) - za * DEVPARAMS::GridWGhost[0];
    x[3] = borderid;
    break;
  }
  idx = x[0] + DEVPARAMS::GridWGhost[0] *
                   (x[1] + DEVPARAMS::GridWGhost[1] *
                               (x[2] + x[3] * DEVPARAMS::GridWGhost[2]));
  idx = idx / 2 + oddbit * DEVPARAMS::HalfVolumeG;
  return idx;
}

template <bool UseTex, ArrayType atype, class Real>
__global__ void PackUnpack_Ghost_Gauge(complex *array, complex *arraypack,
                                       int facesize, bool pack, int borderid,
                                       int faceid, int dir, int oddbit) {
  int id = INDEX1D();
  if (id < facesize) {
    int idx = LatticeFaceIndex00(id, oddbit, faceid, borderid);
    if (pack) {
      msun link = GAUGE_LOAD<UseTex, atype, Real>(
          array, idx + dir * DEVPARAMS::VolumeG, DEVPARAMS::VolumeG * 4);
      GAUGE_SAVE<atype, Real>(arraypack, link, id, facesize);
    } else {
      msun link = GAUGE_LOAD<UseTex, atype, Real>(arraypack, id, facesize);
      GAUGE_SAVE<atype, Real>(array, link, idx + dir * DEVPARAMS::VolumeG,
                              DEVPARAMS::VolumeG * 4);
    }
  }
}

template <class Real>
void CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE(
    dim3 nblocks, dim3 threads, gauge _pgauge, complex *arraypack, int size,
    bool pack, int borderid, int faceid, bool usetex, int dir, int parity,
    cudaStream_t *streamid) {
  typedef void (*TFuncPtr)(complex *, complex *, int, bool, int, int, int, int);
  TFuncPtr ptr = NULL;
  if (usetex) {
#if (NCOLORS == 3)
    if (_pgauge.Type() == SOA)
      ptr = &PackUnpack_Ghost_Gauge<true, SOA, Real>;
    if (_pgauge.Type() == SOA12)
      ptr = &PackUnpack_Ghost_Gauge<true, SOA12, Real>;
    if (_pgauge.Type() == SOA8)
      ptr = &PackUnpack_Ghost_Gauge<true, SOA8, Real>;
#else
    ptr = &PackUnpack_Ghost_Gauge<true, SOA, Real>;
#endif
  } else {
#if (NCOLORS == 3)
    if (_pgauge.Type() == SOA)
      ptr = &PackUnpack_Ghost_Gauge<false, SOA, Real>;
    if (_pgauge.Type() == SOA12)
      ptr = &PackUnpack_Ghost_Gauge<false, SOA12, Real>;
    if (_pgauge.Type() == SOA8)
      ptr = &PackUnpack_Ghost_Gauge<false, SOA8, Real>;
#else
    ptr = &PackUnpack_Ghost_Gauge<false, SOA, Real>;
#endif
  }
  if (ptr != NULL)
    ptr<<<nblocks, threads, 0, *streamid>>>(
        _pgauge.GetPtr(), arraypack, size, pack, borderid, faceid, dir, parity);
  else {
    COUT << "Function type not found.\tExiting..." << std::endl;
    exit(0);
  }
}
template void CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<float>(
    dim3 nblocks, dim3 threads, gauges _pgauge, complexs *arraypack, int size,
    bool pack, int borderid, int faceid, bool usetex, int dir, int parity,
    cudaStream_t *streamid);
template void CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<double>(
    dim3 nblocks, dim3 threads, gauged _pgauge, complexd *arraypack, int size,
    bool pack, int borderid, int faceid, bool usetex, int dir, int parity,
    cudaStream_t *streamid);

template <class Real>
void Exchange_gauge_border_links_gauge(gauge _pgauge, int dir, int parity,
                                       bool all_radius_border) {
  AllocateTempBuffersAndStreams<Real>();
  GAUGE_TEXTURE(_pgauge.GetPtr(), true);

  dim3 threads = dim3(128, 1, 1);
  CUDA_SAFE_DEVICE_SYNC();

  int artype = GetIdFromNumElems(_pgauge.getNumElems());

  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {

    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);

    int nradiusbordertoupdate = 1;
    if (all_radius_border)
      nradiusbordertoupdate = PARAMS::Border[PARAMS::FaceId[fc]];
    for (int border = 0; border < nradiusbordertoupdate; border++) {

      MPI_CHECK(MPI_Start(&top_recv_request_border[artype][fc]));
      MPI_CHECK(MPI_Start(&bot_recv_request_border[artype][fc]));

      // Pack top face border links
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
          blocks, threads, _pgauge,
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc], facesize,
          true,
          PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
              PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
          PARAMS::FaceId[fc], PARAMS::UseTex, dir, parity, &exchangeStream[0]);

      // Pack top face border links
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
          blocks, threads, _pgauge,
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc] +
              ncopyelemshalf,
          facesize, true, PARAMS::Border[PARAMS::FaceId[fc]] + border,
          PARAMS::FaceId[fc], PARAMS::UseTex, dir, parity, &exchangeStream[1]);

#ifndef MPI_GPU_DIRECT
      // D2H top face border links
      CUDA_SAFE_CALL(cudaMemcpyAsync(
          reinterpret_cast<complex *>(sendbuff_cpu) + offsetptr[fc],
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc],
          ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
          exchangeStream[0]));
      CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex *>(sendbuff_cpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     reinterpret_cast<complex *>(sendbuff_gpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     ncopyelemshalf * sizeof(complex),
                                     cudaMemcpyDefault, exchangeStream[1]));
#endif
      // MPI transfers

      CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[0]));
      MPI_CHECK(MPI_Start(&top_send_request_border[artype][fc]));

      CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[1]));
      MPI_CHECK(MPI_Start(&bot_send_request_border[artype][fc]));

      MPI_CHECK(
          MPI_Wait(&top_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#ifndef MPI_GPU_DIRECT
      CUDA_SAFE_CALL(cudaMemcpyAsync(
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc],
          reinterpret_cast<complex *>(recvbuff_cpu) + offsetptr[fc],
          ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
          exchangeStream[0]));
      MPI_CHECK(
          MPI_Wait(&bot_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
      CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex *>(recvbuff_gpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     reinterpret_cast<complex *>(recvbuff_cpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     ncopyelemshalf * sizeof(complex),
                                     cudaMemcpyDefault, exchangeStream[1]));
#endif
      // Unpack ghost links

      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
          blocks, threads, _pgauge,
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc], facesize,
          false, PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
          PARAMS::FaceId[fc], false, dir, parity, &exchangeStream[0]);
#ifdef MPI_GPU_DIRECT
      MPI_CHECK(
          MPI_Wait(&bot_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#endif
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
          blocks, threads, _pgauge,
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc] +
              ncopyelemshalf,
          facesize, false,
          PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
              PARAMS::Border[PARAMS::FaceId[fc]] + border,
          PARAMS::FaceId[fc], false, dir, parity, &exchangeStream[1]);

      MPI_CHECK(
          MPI_Wait(&top_send_request_border[artype][fc], MPI_STATUS_IGNORE));
      MPI_CHECK(
          MPI_Wait(&bot_send_request_border[artype][fc], MPI_STATUS_IGNORE));
      CUDA_SAFE_DEVICE_SYNC();
    }
  }
}
template void Exchange_gauge_border_links_gauge<float>(gauges _pgauge, int dir,
                                                       int parity,
                                                       bool all_radius_border);
template void Exchange_gauge_border_links_gauge<double>(gauged _pgauge, int dir,
                                                        int parity,
                                                        bool all_radius_border);

template <class Real>
void Exchange_gauge_topborder_links_gauge(gauge _pgauge, int dir, int parity,
                                          bool all_radius_border) {

  AllocateTempBuffersAndStreams<Real>();
  GAUGE_TEXTURE(_pgauge.GetPtr(), true);

  dim3 threads = dim3(128, 1, 1);
  CUDA_SAFE_DEVICE_SYNC();
  int artype = GetIdFromNumElems(_pgauge.getNumElems());

  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {

    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);

    int nradiusbordertoupdate = 1;
    if (all_radius_border)
      nradiusbordertoupdate = PARAMS::Border[PARAMS::FaceId[fc]];
    for (int border = 0; border < nradiusbordertoupdate; border++) {

      MPI_CHECK(MPI_Start(&top_recv_request_border[artype][fc]));

      // Pack top face border links
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
          blocks, threads, _pgauge,
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc], facesize,
          true,
          PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
              PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
          PARAMS::FaceId[fc], PARAMS::UseTex, dir, parity, &exchangeStream[0]);

#ifndef MPI_GPU_DIRECT
      // D2H top face border links
      CUDA_SAFE_CALL(cudaMemcpyAsync(
          reinterpret_cast<complex *>(sendbuff_cpu) + offsetptr[fc],
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc],
          ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
          exchangeStream[0]));
#endif
      // MPI transfers

      CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[0]));
      MPI_CHECK(MPI_Start(&top_send_request_border[artype][fc]));

      MPI_CHECK(
          MPI_Wait(&top_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#ifndef MPI_GPU_DIRECT
      CUDA_SAFE_CALL(cudaMemcpyAsync(
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc],
          reinterpret_cast<complex *>(recvbuff_cpu) + offsetptr[fc],
          ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
          exchangeStream[0]));
#endif
      // Unpack ghost links

      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
          blocks, threads, _pgauge,
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc], facesize,
          false, PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
          PARAMS::FaceId[fc], false, dir, parity, &exchangeStream[0]);

      MPI_CHECK(
          MPI_Wait(&top_send_request_border[artype][fc], MPI_STATUS_IGNORE));
      CUDA_SAFE_DEVICE_SYNC();
    }
  }
}

/*

AllocateTempBuffersAndStreams<Real>(_pgauge.getNumElems());
        dim3 threads = dim3(128, 1, 1);
        CUDA_SAFE_DEVICE_SYNC( );
        for(int fc = 0; fc < PARAMS::NActiveFaces; fc++){
                int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
                int ncopyelemshalf = facesize * _pgauge.getNumElems();
                        dim3 blocks = GetBlockDim(threads.x, facesize);
                //Pack top face border links
                CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(blocks,
threads, _pgauge, \
                        reinterpret_cast<complex*>(sendbuff_gpu) +
offsetptr[fc], \
                        facesize, true, \
                        PARAMS::GridWGhost[PARAMS::FaceId[fc]]-PARAMS::Border[PARAMS::FaceId[fc]]
- 1,\ PARAMS::FaceId[fc], PARAMS::UseTex, dir, parity, &exchangeStream[0]);
                #ifndef  MPI_GPU_DIRECT
                //D2H top face border links
                CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex*>(sendbuff_cpu)
+ offsetptr[fc], \
                        reinterpret_cast<complex*>(sendbuff_gpu) +
offsetptr[fc], \ ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
exchangeStream[0])); #endif
                //MPI transfers
                CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[0]));
                #ifndef  MPI_GPU_DIRECT
                MPI_CHECK(MPI_Isend(reinterpret_cast<complex*>(sendbuff_cpu) +
offsetptr[fc], ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdRight[fc], fc, MPI_COMM_WORLD,
&send_request_border[0]));
                MPI_CHECK(MPI_Irecv(reinterpret_cast<complex*>(recvbuff_cpu) +
offsetptr[fc], ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdLeft[fc], fc, MPI_COMM_WORLD,
&recv_request_border[0])); #else
                MPI_CHECK(MPI_Isend(reinterpret_cast<complex*>(sendbuff_gpu) +
offsetptr[fc], ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdRight[fc], fc, MPI_COMM_WORLD,
&send_request_border[0]));
                MPI_CHECK(MPI_Irecv(reinterpret_cast<complex*>(recvbuff_gpu) +
offsetptr[fc], ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdLeft[fc], fc, MPI_COMM_WORLD,
&recv_request_border[0])); #endif #ifndef  MPI_GPU_DIRECT
                //H2D top face border links
                        MPI_CHECK( MPI_Wait( &recv_request_border[0],
&MPI_StatuS ) );
                        CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex*>(recvbuff_gpu)
+ offsetptr[fc], \
                                reinterpret_cast<complex*>(recvbuff_cpu) +
offsetptr[fc], \ ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
exchangeStream[0])); #endif
                //Unpack ghost links

                #ifdef  MPI_GPU_DIRECT
                MPI_CHECK( MPI_Wait( &recv_request_border[0], &MPI_StatuS ) );
                #endif
                CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(blocks,
threads, _pgauge, \
                        reinterpret_cast<complex*>(recvbuff_gpu) +
offsetptr[fc], \
                        facesize, false, PARAMS::Border[PARAMS::FaceId[fc]]-1,\
                        PARAMS::FaceId[fc], false, dir, parity,
&exchangeStream[0]);

                MPI_CHECK( MPI_Wait( &send_request_border[0], &MPI_StatuS ) );
                CUDA_SAFE_DEVICE_SYNC( );
        }
//FreeTempBuffersAndStreams();
}*/
template void
Exchange_gauge_topborder_links_gauge<float>(gauges _pgauge, int dir, int parity,
                                            bool all_radius_border);
template void Exchange_gauge_topborder_links_gauge<double>(
    gauged _pgauge, int dir, int parity, bool all_radius_border);

/*
//If radius border > 1, then only exchange the interior border, 1 layer only
// to update all borders for radius border > 1 use
Exchange_gauge_border_links_gauge_ template<class Real> void
Exchange_gauge_border_links_gauge(gauge _pgauge, int dir, int parity){
        AllocateTempBuffersAndStreams<Real>(_pgauge.getNumElems());
        dim3 threads = dim3(128, 1, 1);
        CUDA_SAFE_DEVICE_SYNC( );
        for(int fc = 0; fc < PARAMS::NActiveFaces; fc++){

                int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
                int ncopyelemshalf = facesize * _pgauge.getNumElems();
                dim3 blocks = GetBlockDim(threads.x, facesize);
                //Pack top face border links
                CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(blocks,
threads, _pgauge, \
                        reinterpret_cast<complex*>(sendbuff_gpu) +
offsetptr[fc], \
                        facesize, true, \
                        PARAMS::GridWGhost[PARAMS::FaceId[fc]]-PARAMS::Border[PARAMS::FaceId[fc]]
- 1,\ PARAMS::FaceId[fc], PARAMS::UseTex, dir, parity, &exchangeStream[0]);
                //Pack bottom face border links
                CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(blocks,
threads, _pgauge, \
                        reinterpret_cast<complex*>(sendbuff_gpu) + offsetptr[fc]
+ ncopyelemshalf, \
                        facesize, true, \
                        PARAMS::Border[PARAMS::FaceId[fc]], \
                        PARAMS::FaceId[fc], PARAMS::UseTex, dir, parity,
&exchangeStream[1]); #ifndef  MPI_GPU_DIRECT
                //D2H top face border links
                CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex*>(sendbuff_cpu)
+ offsetptr[fc], \
                        reinterpret_cast<complex*>(sendbuff_gpu) +
offsetptr[fc], \ ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
exchangeStream[0]));
                //H2D bottom face border links
                CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex*>(sendbuff_cpu)
+ offsetptr[fc] + ncopyelemshalf, \
                        reinterpret_cast<complex*>(sendbuff_gpu) + offsetptr[fc]
+ ncopyelemshalf, \ ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
exchangeStream[1])); #endif
                //MPI transfers
                CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[0]));
                MPI_CHECK(MPI_Isend(reinterpret_cast<complex*>(sendbuff) +
offsetptr[fc], ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdRight[fc], fc, MPI_COMM_WORLD,
&send_request_border[0]));
                MPI_CHECK(MPI_Irecv(reinterpret_cast<complex*>(recvbuff) +
offsetptr[fc], ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdLeft[fc], fc, MPI_COMM_WORLD,
&recv_request_border[0]));

                CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[1]));
                MPI_CHECK(MPI_Isend(reinterpret_cast<complex*>(sendbuff) +
offsetptr[fc] + ncopyelemshalf, \
                        ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdLeft[fc], fc, MPI_COMM_WORLD,
&send_request_border[1]));
                MPI_CHECK(MPI_Irecv(reinterpret_cast<complex*>(recvbuff) +
offsetptr[fc] + ncopyelemshalf, \
                        ncopyelemshalf, mpi_datatype<complex>(), \
                        PARAMS::NodeIdRight[fc], fc, MPI_COMM_WORLD,
&recv_request_border[1]));


            #ifndef  MPI_GPU_DIRECT
                //H2D received top face border links
                MPI_CHECK( MPI_Wait( &recv_request_border[0], &MPI_StatuS ) );
                CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex*>(recvbuff_gpu)
+ offsetptr[fc], \
                        reinterpret_cast<complex*>(recvbuff_cpu) +
offsetptr[fc], \ ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
exchangeStream[0]));
                //H2D received bottom face border links
                MPI_CHECK( MPI_Wait( &recv_request_border[1], &MPI_StatuS ) );
                CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex*>(recvbuff_gpu)
+ offsetptr[fc] + ncopyelemshalf, \
                        reinterpret_cast<complex*>(recvbuff_cpu) + offsetptr[fc]
+ ncopyelemshalf, \ ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
exchangeStream[1])); #endif
                //Unpack ghost links

        #ifdef  MPI_GPU_DIRECT
                MPI_CHECK( MPI_Wait( &recv_request_border[0], &MPI_StatuS ) );
        #endif
                CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(blocks,
threads, _pgauge, \
                        reinterpret_cast<complex*>(recvbuff_gpu) +
offsetptr[fc], \
                        facesize, false, PARAMS::Border[PARAMS::FaceId[fc]]-1,\
                        PARAMS::FaceId[fc], false, dir, parity,
&exchangeStream[0]); #ifdef  MPI_GPU_DIRECT MPI_CHECK( MPI_Wait(
&recv_request_border[1], &MPI_StatuS ) ); #endif
                CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(blocks,
threads, _pgauge, \
                        reinterpret_cast<complex*>(recvbuff_gpu) + offsetptr[fc]
+ ncopyelemshalf, \
                        facesize, false,
PARAMS::GridWGhost[PARAMS::FaceId[fc]]-PARAMS::Border[PARAMS::FaceId[fc]],\
                        PARAMS::FaceId[fc], false, dir, parity,
&exchangeStream[1]);

                MPI_CHECK( MPI_Wait( &send_request_border[0], &MPI_StatuS ) );
                MPI_CHECK( MPI_Wait( &send_request_border[1], &MPI_StatuS ) );
                CUDA_SAFE_DEVICE_SYNC( );



        }
//FreeTempBuffersAndStreams()
}
template
void  Exchange_gauge_border_links_gauge<float>(gauges _pgauge, int dir, int
parity); template void  Exchange_gauge_border_links_gauge<double>(gauged
_pgauge, int dir, int parity);
;*/

template <class Real>
void StartExchange_gauge_fix_links_gauge(gauge _pgauge, int parity) {
  AllocateTempBuffersAndStreams<Real>();
  GAUGE_TEXTURE(_pgauge.GetPtr(), true);

  // for gauge fix only needs to exchange links along partitioned faces

  dim3 threads = dim3(128, 1, 1);
  int artype = GetIdFromNumElems(_pgauge.getNumElems());

  int border = 0;
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);
    // Pack top face border links
    CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
        blocks, threads, _pgauge,
        reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc], facesize,
        true,
        PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
            PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
        PARAMS::FaceId[fc], PARAMS::UseTex, PARAMS::FaceId[fc], parity,
        &exchangeStream[fc]);
    MPI_CHECK(MPI_Start(&top_recv_request_border[artype][fc]));
  }
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);
    // Pack top face border links
    CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
        blocks, threads, _pgauge,
        reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc] +
            ncopyelemshalf,
        facesize, true, PARAMS::Border[PARAMS::FaceId[fc]] - 1,
        PARAMS::FaceId[fc], PARAMS::UseTex, PARAMS::FaceId[fc], 1 - parity,
        &exchangeStream[4 + fc]);
    MPI_CHECK(MPI_Start(&bot_recv_request_border[artype][fc]));
  }

#ifdef MPI_GPU_DIRECT
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[fc]));
    MPI_CHECK(MPI_Start(&top_send_request_border[artype][fc]));
    CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[4 + fc]));
    MPI_CHECK(MPI_Start(&bot_send_request_border[artype][fc]));
  }
#else
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    // D2H top face border links
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        reinterpret_cast<complex *>(sendbuff_cpu) + offsetptr[fc],
        reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc],
        ncopyelemshalf * sizeof(complex), cudaMemcpyDeviceToHost,
        exchangeStream[fc]));
  }
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex *>(sendbuff_cpu) +
                                       offsetptr[fc] + ncopyelemshalf,
                                   reinterpret_cast<complex *>(sendbuff_gpu) +
                                       offsetptr[fc] + ncopyelemshalf,
                                   ncopyelemshalf * sizeof(complex),
                                   cudaMemcpyDeviceToHost,
                                   exchangeStream[4 + fc]));
  }
#endif
}
template void StartExchange_gauge_fix_links_gauge<float>(gauges _pgauge,
                                                         int parity);
template void StartExchange_gauge_fix_links_gauge<double>(gauged _pgauge,
                                                          int parity);

template <class Real>
void EndExchange_gauge_fix_links_gauge(gauge _pgauge, int parity) {

  dim3 threads = dim3(128, 1, 1);
  int artype = GetIdFromNumElems(_pgauge.getNumElems());
  int border = 0;

#ifndef MPI_GPU_DIRECT
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[fc]));
    MPI_CHECK(MPI_Start(&top_send_request_border[artype][fc]));
    CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[4 + fc]));
    MPI_CHECK(MPI_Start(&bot_send_request_border[artype][fc]));
  }
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    MPI_CHECK(
        MPI_Wait(&top_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc],
        reinterpret_cast<complex *>(recvbuff_cpu) + offsetptr[fc],
        ncopyelemshalf * sizeof(complex), cudaMemcpyHostToDevice,
        exchangeStream[fc]));
  }
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    MPI_CHECK(
        MPI_Wait(&bot_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex *>(recvbuff_gpu) +
                                       offsetptr[fc] + ncopyelemshalf,
                                   reinterpret_cast<complex *>(recvbuff_cpu) +
                                       offsetptr[fc] + ncopyelemshalf,
                                   ncopyelemshalf * sizeof(complex),
                                   cudaMemcpyHostToDevice,
                                   exchangeStream[4 + fc]));
  }
#endif
  // Unpack ghost links

  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
#ifdef MPI_GPU_DIRECT
    MPI_CHECK(
        MPI_Wait(&top_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#endif
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);
    CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
        blocks, threads, _pgauge,
        reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc], facesize,
        false, PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
        PARAMS::FaceId[fc], false, PARAMS::FaceId[fc], parity,
        &exchangeStream[fc]);
  }
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
#ifdef MPI_GPU_DIRECT
    MPI_CHECK(
        MPI_Wait(&bot_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#endif
    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * _pgauge.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);
    CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GAUGE<Real>(
        blocks, threads, _pgauge,
        reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc] +
            ncopyelemshalf,
        facesize, false,
        PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
            PARAMS::Border[PARAMS::FaceId[fc]] + border - 1,
        PARAMS::FaceId[fc], false, PARAMS::FaceId[fc], 1 - parity,
        &exchangeStream[4 + fc]);
  }
  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
    MPI_CHECK(
        MPI_Wait(&top_send_request_border[artype][fc], MPI_STATUS_IGNORE));
    MPI_CHECK(
        MPI_Wait(&bot_send_request_border[artype][fc], MPI_STATUS_IGNORE));
  }
}
template void EndExchange_gauge_fix_links_gauge<float>(gauges _pgauge,
                                                       int parity);
template void EndExchange_gauge_fix_links_gauge<double>(gauged _pgauge,
                                                        int parity);

template <bool UseTex, ArrayType atype, class Real>
__global__ void
PackUnpack_Ghost_GX(complex *array, complex *arraypack, int facesize, bool pack,
                    int borderid, int faceid, int dir, int oddbit, int offset) {
  int id = INDEX1D();
  if (id < facesize) {
    int idx = LatticeFaceIndex00(id, oddbit, faceid, borderid);
    if (pack) {
      msun link = GAUGE_LOAD<UseTex, atype, Real>(
          array, idx + dir * DEVPARAMS::VolumeG, offset);
      GAUGE_SAVE<atype, Real>(arraypack, link, id, facesize);
    } else {
      msun link = GAUGE_LOAD<UseTex, atype, Real>(arraypack, id, facesize);
      GAUGE_SAVE<atype, Real>(array, link, idx + dir * DEVPARAMS::VolumeG,
                              offset);
    }
  }
}

template <class Real>
void CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX(
    dim3 nblocks, dim3 threads, gauge array, complex *arraypack, int size,
    bool pack, int borderid, int faceid, bool usetex, int dir, int parity,
    int offset, cudaStream_t *streamid) {
  typedef void (*TFuncPtr)(complex *, complex *, int, bool, int, int, int, int,
                           int);
  TFuncPtr ptr = NULL;
  if (usetex) {
#if (NCOLORS == 3)
    if (array.Type() == SOA)
      ptr = &PackUnpack_Ghost_GX<true, SOA, Real>;
    if (array.Type() == SOA12)
      ptr = &PackUnpack_Ghost_GX<true, SOA12, Real>;
    if (array.Type() == SOA8)
      ptr = &PackUnpack_Ghost_GX<true, SOA8, Real>;
#else
    ptr = &PackUnpack_Ghost_GX<true, SOA, Real>;
#endif
  } else {
#if (NCOLORS == 3)
    if (array.Type() == SOA)
      ptr = &PackUnpack_Ghost_GX<false, SOA, Real>;
    if (array.Type() == SOA12)
      ptr = &PackUnpack_Ghost_GX<false, SOA12, Real>;
    if (array.Type() == SOA8)
      ptr = &PackUnpack_Ghost_GX<false, SOA8, Real>;
#else
    ptr = &PackUnpack_Ghost_GX<false, SOA, Real>;
#endif
  }
  if (ptr != NULL)
    ptr<<<nblocks, threads, 0, *streamid>>>(array.GetPtr(), arraypack, size,
                                            pack, borderid, faceid, dir, parity,
                                            offset);
  else {
    COUT << "Function type not found.\tExiting..." << std::endl;
    exit(0);
  }
}
template void CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX<float>(
    dim3 nblocks, dim3 threads, gauges array, complexs *arraypack, int size,
    bool pack, int borderid, int faceid, bool usetex, int dir, int parity,
    int offset, cudaStream_t *streamid);
template void CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX<double>(
    dim3 nblocks, dim3 threads, gauged array, complexd *arraypack, int size,
    bool pack, int borderid, int faceid, bool usetex, int dir, int parity,
    int offset, cudaStream_t *streamid);

template <class Real>
void Exchange_gauge_border_links(gauge array, int dir, int parity,
                                 bool all_radius_border) {

  AllocateTempBuffersAndStreams<Real>();

  dim3 threads = dim3(128, 1, 1);
  CUDA_SAFE_DEVICE_SYNC();

  int artype = GetIdFromNumElems(array.getNumElems());

  for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {

    int facesize = PARAMS::FaceSizeG[PARAMS::FaceId[fc]] >> 1;
    int ncopyelemshalf = facesize * array.getNumElems();
    dim3 blocks = GetBlockDim(threads.x, facesize);

    int nradiusbordertoupdate = 1;
    if (all_radius_border)
      nradiusbordertoupdate = PARAMS::Border[PARAMS::FaceId[fc]];
    for (int border = 0; border < nradiusbordertoupdate; border++) {

      MPI_CHECK(MPI_Start(&top_recv_request_border[artype][fc]));
      MPI_CHECK(MPI_Start(&bot_recv_request_border[artype][fc]));

      // Pack top face border links
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX<Real>(
          blocks, threads, array,
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc], facesize,
          true,
          PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
              PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
          PARAMS::FaceId[fc], false, dir, parity, array.Size(),
          &exchangeStream[0]);

      // Pack top face border links
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX<Real>(
          blocks, threads, array,
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc] +
              ncopyelemshalf,
          facesize, true, PARAMS::Border[PARAMS::FaceId[fc]] + border,
          PARAMS::FaceId[fc], false, dir, parity, array.Size(),
          &exchangeStream[1]);

#ifndef MPI_GPU_DIRECT
      // D2H top face border links
      CUDA_SAFE_CALL(cudaMemcpyAsync(
          reinterpret_cast<complex *>(sendbuff_cpu) + offsetptr[fc],
          reinterpret_cast<complex *>(sendbuff_gpu) + offsetptr[fc],
          ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
          exchangeStream[0]));
      CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex *>(sendbuff_cpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     reinterpret_cast<complex *>(sendbuff_gpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     ncopyelemshalf * sizeof(complex),
                                     cudaMemcpyDefault, exchangeStream[1]));
#endif
      // MPI transfers

      CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[0]));
      MPI_CHECK(MPI_Start(&top_send_request_border[artype][fc]));

      CUDA_SAFE_CALL(cudaStreamSynchronize(exchangeStream[1]));
      MPI_CHECK(MPI_Start(&bot_send_request_border[artype][fc]));

      MPI_CHECK(
          MPI_Wait(&top_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#ifndef MPI_GPU_DIRECT
      CUDA_SAFE_CALL(cudaMemcpyAsync(
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc],
          reinterpret_cast<complex *>(recvbuff_cpu) + offsetptr[fc],
          ncopyelemshalf * sizeof(complex), cudaMemcpyDefault,
          exchangeStream[0]));
      MPI_CHECK(
          MPI_Wait(&bot_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
      CUDA_SAFE_CALL(cudaMemcpyAsync(reinterpret_cast<complex *>(recvbuff_gpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     reinterpret_cast<complex *>(recvbuff_cpu) +
                                         offsetptr[fc] + ncopyelemshalf,
                                     ncopyelemshalf * sizeof(complex),
                                     cudaMemcpyDefault, exchangeStream[1]));
#endif
      // Unpack ghost links

      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX<Real>(
          blocks, threads, array,
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc], facesize,
          false, PARAMS::Border[PARAMS::FaceId[fc]] - 1 - border,
          PARAMS::FaceId[fc], false, dir, parity, array.Size(),
          &exchangeStream[0]);
#ifdef MPI_GPU_DIRECT
      MPI_CHECK(
          MPI_Wait(&bot_recv_request_border[artype][fc], MPI_STATUS_IGNORE));
#endif
      CALL_PACK_UNPACK_BORDERGHOST_LINKS_FACE_GX<Real>(
          blocks, threads, array,
          reinterpret_cast<complex *>(recvbuff_gpu) + offsetptr[fc] +
              ncopyelemshalf,
          facesize, false,
          PARAMS::GridWGhost[PARAMS::FaceId[fc]] -
              PARAMS::Border[PARAMS::FaceId[fc]] + border,
          PARAMS::FaceId[fc], false, dir, parity, array.Size(),
          &exchangeStream[1]);

      MPI_CHECK(
          MPI_Wait(&top_send_request_border[artype][fc], MPI_STATUS_IGNORE));
      MPI_CHECK(
          MPI_Wait(&bot_send_request_border[artype][fc], MPI_STATUS_IGNORE));
      CUDA_SAFE_DEVICE_SYNC();
    }
  }
}
template void Exchange_gauge_border_links(gauges array, int dir, int parity,
                                          bool all_radius_border);
template void Exchange_gauge_border_links(gauged array, int dir, int parity,
                                          bool all_radius_border);

#endif

} // namespace CULQCD
