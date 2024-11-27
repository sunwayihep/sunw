
#ifndef REUNITLINK_H
#define REUNITLINK_H

#include <complex.h>
#include <matrixsun.h>

namespace CULQCD {

/**
    @brief Reunitarize link. Only works for SU(Nc) with Nc > 2.
    @param U link to be reunitarized and updated.
*/
template <class Real> __host__ __device__ inline void reunit_link(msun *U) {
#if (NCOLORS == 3)
  Real t1 = 0.0;
  complex t2 = complex::zero();
// first normalize first row
// sum of squares of row
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    t1 += (*U)(0, c).abs2();
  t1 = (Real)1.0 / sqrt(t1);
// used to normalize row
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    (*U)(0, c) *= t1;
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    t2 += (*U)(0, c).conj() * (*U)(1, c);
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    (*U)(1, c) -= t2 * (*U)(0, c);
  // normalize second row
  // sum of squares of row
  t1 = 0.0;
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    t1 += (*U)(1, c).abs2();
  t1 = (float)1.0 / sqrt(t1);
// used to normalize row
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    (*U)(1, c) *= t1;
  // Reconstruct lat row
  (*U)(2, 0) = ((*U)(0, 1) * (*U)(1, 2) - (*U)(0, 2) * (*U)(1, 1)).conj();
  (*U)(2, 1) = ((*U)(0, 2) * (*U)(1, 0) - (*U)(0, 0) * (*U)(1, 2)).conj();
  (*U)(2, 2) = ((*U)(0, 0) * (*U)(1, 1) - (*U)(0, 1) * (*U)(1, 0)).conj();
///////////////////////////////////////////////////////////////////////////////
#elif (NCOLORS > 3)
  ////////////////////////////////// NCOLORS > 3 ////////////////////////////
  Real t1;
  complex t2;
  t1 = 0.0;
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    t1 += U->e[c][0].abs2();
  t1 = 1.0 / sqrt(t1);
#pragma unroll
  for (int c = 0; c < NCOLORS; c++)
    U->e[c][0] *= t1;
// Do Gramm-Schmidt on the remaining rows
#pragma unroll
  for (int j = 1; j < NCOLORS; j++) {
    for (int i = 0; i < j; i++) {
      t2 = (~U->e[0][i]) * U->e[0][j];
      for (int c = 1; c < NCOLORS; ++c) {
        t2 += (~U->e[c][i]) * U->e[c][j];
      }
      for (int c = 0; c < NCOLORS; ++c) {
        U->e[c][j] -= t2 * U->e[c][i];
      }
    }
    t1 = 0.0;
    for (int c = 0; c < NCOLORS; ++c)
      t1 += U->e[c][j].abs2();
    t1 = 1.0 / sqrt(t1);

    for (int c = 0; c < NCOLORS; ++c)
      U->e[c][j] *= t1;
  }
  // The determinant
  t2 = U->det();
  // The phase of the determinant
  Real ai = atan2(t2.imag(), t2.real());
  t2 = complex::make_complex(cos(ai), -sin(ai));
  for (int c = 0; c < NCOLORS; ++c)
    U->e[c][NCOLORS - 1] *= t2;
#else
#warning NCOLORS not defined or NCOLORS < 3!
#endif
}
} // namespace CULQCD

#endif
