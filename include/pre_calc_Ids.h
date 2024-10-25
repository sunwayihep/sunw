
#ifndef GAUGEFIXING_EXCHANGE_H
#define GAUGEFIXING_EXCHANGE_H

#include <gaugearray.h>
#include <constants.h>

namespace CULQCD {

#ifdef MULTI_GPU

extern size_t offsetptr[4];
extern size_t offsetptr_ghost[4];
extern int streamsovrnum;
extern cudaStream_t *streamsovr;
extern MPI_Request send_request_border[4];
extern MPI_Request recv_request_border[4];

extern int *faceindices[2];
extern int faceindicessize[2];
extern int *interiorindices[2];
extern int interiorindicessize[2];

inline __device__ int LatticeFaceIndex(int idd, int oddbit, int faceid,
                                       int borderid) {
  int x1, x2, x3, x4, idx, za, xodd;
  switch (faceid) {
  case 0: // X FACE
    za = idd / (DEVPARAMS::Grid[1] / 2);
    x4 = za / DEVPARAMS::Grid[2];
    x3 = za - x4 * DEVPARAMS::Grid[2];
    xodd = (borderid + x3 + x4 + oddbit) & 1;
    x2 = (2 * idd + xodd) - za * DEVPARAMS::Grid[1];
    x1 = borderid;
    idx = x1 + DEVPARAMS::GridWGhost[0] *
                   (x2 + DEVPARAMS::GridWGhost[1] *
                             (x3 + x4 * DEVPARAMS::GridWGhost[2]));
    idx = idx / 2 + oddbit * DEVPARAMS::HalfVolumeG;
    break;
  case 1: // Y FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x4 = za / DEVPARAMS::Grid[2];
    x3 = za - x4 * DEVPARAMS::Grid[2];
    xodd = (borderid + x3 + x4 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x2 = borderid;
    idx = x1 + DEVPARAMS::GridWGhost[0] *
                   (x2 + DEVPARAMS::GridWGhost[1] *
                             (x3 + x4 * DEVPARAMS::GridWGhost[2]));
    idx = idx / 2 + oddbit * DEVPARAMS::HalfVolumeG;
    break;
  case 2: // Z FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x4 = za / DEVPARAMS::Grid[1];
    x2 = za - x4 * DEVPARAMS::Grid[1];
    xodd = (borderid + x2 + x4 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x3 = borderid;
    idx = x1 + DEVPARAMS::GridWGhost[0] *
                   (x2 + DEVPARAMS::GridWGhost[1] *
                             (x3 + x4 * DEVPARAMS::GridWGhost[2]));
    idx = idx / 2 + oddbit * DEVPARAMS::HalfVolumeG;
    break;
  case 3: // T FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x3 = za / DEVPARAMS::Grid[1];
    x2 = za - x3 * DEVPARAMS::Grid[1];
    xodd = (borderid + x2 + x3 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x4 = borderid;
    idx = x1 + DEVPARAMS::GridWGhost[0] *
                   (x2 + DEVPARAMS::GridWGhost[1] *
                             (x3 + x4 * DEVPARAMS::GridWGhost[2]));
    idx = idx / 2 + oddbit * DEVPARAMS::HalfVolumeG;
    break;
  }
  return idx;
}

inline __device__ void LatticeFaceIndex(int &x1, int &x2, int &x3, int &x4,
                                        int idd, int oddbit, int faceid,
                                        int borderid) {
  int za, xodd;
  switch (faceid) {
  case 0: // X FACE
    za = idd / (DEVPARAMS::Grid[1] / 2);
    x4 = za / DEVPARAMS::Grid[2];
    x3 = za - x4 * DEVPARAMS::Grid[2];
    xodd = (borderid + x3 + x4 + oddbit) & 1;
    x2 = (2 * idd + xodd) - za * DEVPARAMS::Grid[1];
    x1 = borderid;
    break;
  case 1: // Y FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x4 = za / DEVPARAMS::Grid[2];
    x3 = za - x4 * DEVPARAMS::Grid[2];
    xodd = (borderid + x3 + x4 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x2 = borderid;
    break;
  case 2: // Z FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x4 = za / DEVPARAMS::Grid[1];
    x2 = za - x4 * DEVPARAMS::Grid[1];
    xodd = (borderid + x2 + x4 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x3 = borderid;
    break;
  case 3: // T FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x3 = za / DEVPARAMS::Grid[1];
    x2 = za - x3 * DEVPARAMS::Grid[1];
    xodd = (borderid + x2 + x3 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x4 = borderid;
    break;
  }
}

/*template<class Real>
void CALL_PACK_UNPACK_BORDER_LINKS_FACE_GAUGE(dim3 nblocks, dim3 threads, gauge
_pgauge, complex *arraypack, \
  int size, int oddbit, bool pack, int borderid, int faceid, bool usetex,
cudaStream_t *streamid);


template<class Real>
void  Exchange_gauge_topborder_links_gauge(gauge _pgauge, int oddbit);



template<class Real>
void CALL_PACK_UNPACK_BORDER_LINKS_FACE_GAUGE(dim3 nblocks, dim3 threads, gauge
_pgauge, complex *arraypack, \
  int size, bool pack, int borderid, int faceid, bool usetex, cudaStream_t
*streamid);


template<class Real>
void  Exchange_gauge_topborder_links_gauge(gauge _pgauge);


*/

template <class Real>
void CALL_PACK_UNPACK_BORDER_LINKS_FACE_GAUGE(dim3 nblocks, dim3 threads,
                                              gauge _pgauge, complex *arraypack,
                                              int size, bool pack, int borderid,
                                              int faceid, bool usetex,
                                              cudaStream_t *streamid);

template <class Real> void Exchange_gauge_topborder_links_gauge(gauge _pgauge);

template <class Real>
void CALL_PACK_UNPACK_BORDER_LINKS_FACE_GAUGE(dim3 nblocks, dim3 threads,
                                              gauge _pgauge, complex *arraypack,
                                              int size, int oddbit, bool pack,
                                              int borderid, int faceid,
                                              bool usetex,
                                              cudaStream_t *streamid);

template <class Real>
void Exchange_gauge_topborder_links_gauge(gauge _pgauge, int oddbit);

template <class Real>
void CALL_PACK_UNPACK_LINKS_FACE_GX(dim3 nblocks, dim3 threads, gauge _pgauge,
                                    complex *arraypack, int size, bool pack,
                                    int borderid, int faceid, bool usetex,
                                    cudaStream_t *streamid);

template <class Real> void Exchange_gauge_bottomborder_links_gx(gauge _pgauge);

template <class Real>
void CALL_PACK_UNPACK_LINKS_FACE_GX(dim3 nblocks, dim3 threads, gauge _pgauge,
                                    complex *arraypack, int size, int oddbit,
                                    bool pack, int borderid, int faceid,
                                    bool usetex, cudaStream_t *streamid);

template <class Real>
void Exchange_gauge_bottomborder_links_gx(gauge _pgauge, int oddbit);

#endif
}
#endif
