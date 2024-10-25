
#ifndef RECONSTRUCT8P_H
#define RECONSTRUCT8P_H

#include <cuda_common.h>
#include <complex.h>
#include <matrixsun.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using namespace std;

namespace CULQCD {

template <class Real> __host__ __device__ inline void reconstruct12p(msu3 &A) {
  A.e[2][0] = ~(A.e[0][1] * A.e[1][2] - A.e[0][2] * A.e[1][1]);
  A.e[2][1] = ~(A.e[0][2] * A.e[1][0] - A.e[0][0] * A.e[1][2]);
  A.e[2][2] = ~(A.e[0][0] * A.e[1][1] - A.e[0][1] * A.e[1][0]);
}

template <class Real>
__host__ __device__ inline void reconstruct12p_dagger(msu3 &A) {
  A.e[0][2] = ~(A.e[1][0] * A.e[2][1] - A.e[2][0] * A.e[1][1]);
  A.e[1][2] = ~(A.e[2][0] * A.e[0][1] - A.e[0][0] * A.e[2][1]);
  A.e[2][2] = ~(A.e[0][0] * A.e[1][1] - A.e[1][0] * A.e[0][1]);
}

/**
        @brief Reconstruction SU(3) complex matrix using 8 real parameters
*/
template <class Real>
__host__ __device__ inline void reconstruct8p(msu3 &m, complex theta) {
  /*
  Reconstruct from:
  m.e[0][1] = array[id];
A.e[0][2] = array[id +  offset];
A.e[1][0] = array[id + 2 * offset];
theta = array[id + 3 * offset];
theta was obtained from:
  theta.real() = A.e[0][0].phase();
  theta.imag() = A.e[2][0].phase();
The 8 real parameters are stored as 4 complex numbers
  */
  Real p1, p2, diff;
  complex temp;
  p1 = m.e[0][1].abs2() + m.e[0][2].abs2();
  p2 = 1.0 / p1;
  // reconstruct a1, e00
  diff = 1.0 - p1;
  p1 = sqrt(diff >= 0 ? diff : 0.0);
  m.e[0][0].real() = p1 * cos(theta.real());
  m.e[0][0].imag() = p1 * sin(theta.real());
  // reconstruct c1, e20
  diff = 1.0 - m.e[0][0].abs2() - m.e[1][0].abs2();
  p1 = sqrt(diff >= 0 ? diff : 0.0);
  m.e[2][0].real() = p1 * cos(theta.imag());
  m.e[2][0].imag() = p1 * sin(theta.imag());
  // calculate b2, e11
  m.e[1][1] = (m.e[2][0] * m.e[0][2]).conj();
  m.e[2][2] = m.e[0][0].conj() * m.e[1][0];
  m.e[1][1] += m.e[2][2] * m.e[0][1];
  m.e[1][1] *= -p2;
  // calculate b3, e12
  m.e[1][2] = (m.e[2][0] * m.e[0][1]).conj();
  m.e[1][2] -= m.e[2][2] * m.e[0][2];
  m.e[1][2] *= p2;
  // calculate c2, e21
  m.e[2][1] = (m.e[1][0] * m.e[0][2]).conj();
  temp = m.e[0][0].conj() * m.e[2][0];
  m.e[2][1] -= temp * m.e[0][1];
  m.e[2][1] *= p2;
  // calculate c3, e22
  m.e[2][2] = (m.e[1][0] * m.e[0][1]).conj();
  m.e[2][2] += temp * m.e[0][2];
  m.e[2][2] *= -p2;
}
}

#endif
