

#ifndef PROJECTLINK_H
#define PROJECTLINK_H

#include "complex.h"
#include "device_PHB_OVR.h"
#include "matrixsun.h"
#include "msu2.h"

namespace CULQCD {

template <class Real>
__host__ __device__ inline void project_link(msun &w, msun q, int Nhit,
                                             Real tol) {

  msun action;
  q = q.dagger();
  Real conver, old_tr = 0, new_tr;
  if (tol > 0)
    old_tr = realtraceUVdagger(w, q) / (Real)NCOLORS;
  conver = 1.0;

#if (NCOLORS == 3)
  int nblocks = 3;
#else
  int nblocks = (NCOLORS * (NCOLORS - 1) / 2);
#endif
  /* Do SU(2) hits */
  for (int index1 = 0; index1 < Nhit && conver > tol; index1++) {
    for (int block = 0; block < nblocks; block++) {
      action = w * q;
#if (NCOLORS == 3)
      msu2 r = get_block_su2<Real>(action, block);
      Real k = r.abs();
      if (k == 0.0) {
        k = 1.0;
        r.a0() = 1.0;
      }
      k = (Real)1.0 / k;
      r.a0() *= k;
      r.a1() *= -k;
      r.a2() *= -k;
      r.a3() *= -k;
      ///////////////////////////////////////
      complex a00 = complex::make_complex(r.a0(), r.a3());
      complex a01 = complex::make_complex(r.a2(), r.a1());
      complex a10 = complex::make_complex(-r.a2(), r.a1());
      complex a11 = complex::make_complex(r.a0(), -r.a3());
      block_su2_to_su3<Real>(w, a00, a01, a10, a11, block);
#else
      int2 id = IndexBlock(block);
      msu2 r = get_block_su2<Real>(action, id);
      Real k = r.abs();
      if (k == 0.0) {
        k = 1.0;
        r.a0() = 1.0;
      }
      k = (Real)1.0 / k;
      r.a0() *= k;
      r.a1() *= -k;
      r.a2() *= -k;
      r.a3() *= -k;
      mul_block_sun<Real>(r, w, id);
#endif
    }
    if (tol > 0) {
      new_tr = realtraceUVdagger(w, q) / (Real)NCOLORS;
      conver = (new_tr - old_tr) / old_tr; /* trace always increases */
      old_tr = new_tr;
    }
  } /* hits */
  if (Nhit > 0 && tol > 0 && conver > tol)
    printf("project_link SU(%d): No convergence: conver = %e\n", NCOLORS,
           conver);
}

template <class Real>
__host__ __device__ void SU2project(complex &u11, complex &u12, complex &u21,
                                    complex &u22) {
  Real k =
      (Real)1.0 / sqrt((u11.real() + u22.real()) * (u11.real() + u22.real()) +
                       (u21.imag() + u12.imag()) * (u21.imag() + u12.imag()) +
                       (u12.real() - u21.real()) * (u12.real() - u21.real()) +
                       (u11.imag() - u22.imag()) * (u11.imag() - u22.imag()));

  Real a0, a1, a2, a3;

  a0 = k * (u11.real() + u22.real());
  a1 = k * (u21.imag() + u12.imag());
  a2 = k * (u12.real() - u21.real());
  a3 = k * (u11.imag() - u22.imag());

  u11 = complex::make_complex(a0, a3);
  u12 = complex::make_complex(a2, a1);
  u21 = complex::make_complex(-a2, a1);
  u22 = complex::make_complex(a0, -a3);
}

/// Projects the SU(2) subgroup of a 3x3 matrix onto SU(2)
template <class Real> __host__ __device__ void SU2project(msun &V, int i) {

  switch (i) {
  case 0:
    SU2project<Real>(V.e[0][0], V.e[0][1], V.e[1][0], V.e[1][1]);
    V.e[2][2] = complex::one();
    V.e[0][2] = V.e[1][2] = V.e[2][0] = V.e[2][1] = complex::zero();
    break;
  case 1:
    SU2project<Real>(V.e[1][1], V.e[1][2], V.e[2][1], V.e[2][2]);
    V.e[0][0] = complex::one();
    V.e[0][1] = V.e[0][2] = V.e[1][0] = V.e[2][0] = complex::zero();
    break;
  case 2:
    SU2project<Real>(V.e[0][0], V.e[0][2], V.e[2][0], V.e[2][2]);
    V.e[1][1] = complex::one();
    V.e[0][1] = V.e[1][0] = V.e[1][2] = V.e[2][1] = complex::zero();
    break;
  }
}

//! Projects the matrix onto SU(3) and replaces it by its projection
template <class Real>
__host__ __device__ void SU3project(msun &F, int max_iter) {

  // const int max_iter = 10;
  msun U = msun::unit();
  F = F.dagger();
  for (int i = 0; i < max_iter; ++i) {
    msun V1 = msun::unit(), V2 = msun::unit(), V3 = msun::unit();
    V1 = U * F;
    SU2project<Real>(V1, 0);
    U = V1.dagger() * U;
    V2 = U * F;
    SU2project<Real>(V2, 1);
    U = V2.dagger() * U;
    V3 = U * F;
    SU2project<Real>(V3, 2);
    U = V3.dagger() * U;
  }

  F = U;
}
///////////////////////////////////////////////////////////////
} // namespace CULQCD

#endif
