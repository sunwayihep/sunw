
#ifndef EESU3_H
#define EESU3_H

#include <complex.h>
#include <matrixsun.h>

namespace CULQCD {
//! Exact exponentiation of SU(3) matrix.
/*!
 *  Input: 3x3 anti-Hermitian, traceless matrix iQ
 *  Output: exp(iQ)
 *
 *  Formula for exp(iQ) = f0 + f1*Q + f2*Q*Q is found
 *  in section III of hep-lat/0311018.
 */

template <typename T> __host__ __device__ T where(bool a, T b, T c) {
  if (a)
    return b;
  return c;
}

template <class Real> __host__ __device__ inline void eesu3(msun &iQ) {

  msun Q = timesMinusI(iQ);

  complex f0, f1, f2;

  msun QQ = Q * Q;

  Real c0 = (1.0 / 3.0) * (Q * QQ).realtrace();
  Real c1 = 0.5 * QQ.realtrace();
  Real c0abs = fabs(c0);
  Real c0max = 2.0 * pow((c1 / 3.0), 1.5);
  Real theta = acos(c0abs / c0max);
  Real u = sqrt(c1 / 3.0) * cos(theta / 3.0);
  Real w = sqrt(c1) * sin(theta / 3.0);
  Real uu = u * u;
  Real ww = w * w;
  Real cosu = cos(u);
  Real cosw = cos(w);
  Real sinu = sin(u);
  Real sinw = sin(w);

  // exp(2iu) and exp(-iu)
  complex exp2iu = complex((2 * cosu * cosu - 1), 2 * cosu * sinu);
  complex expmiu = complex(cosu, -sinu);

  bool latboo_c0 = (c0 < 0);
  bool latboo_c1 = (c1 > 1.0e-4);
  bool latboo_w = (fabs(w) > 0.05);

  Real denom = where(latboo_c1, (Real)9.0 * uu - ww, (Real)1.0);

  // xi0 = xi0(w).  Expand xi0 if w is small.
  Real xi0 = where(
      latboo_w, sinw / w,
      (Real)(1.0 - (1.0 / 6.0) * ww *
                       (1.0 - (1.0 / 20.0) * ww * (1.0 - (1.0 / 42.0) * ww))));

  // f_i = f_i(c0, c1). Expand f_i by c1, if c1 is small.
  f0 = where(latboo_c1, (exp2iu * (uu - ww) +
                         expmiu * complex(8.0 * uu * cosw,
                                          2.0 * u * (3.0 * uu + ww) * xi0)) /
                            denom,
             complex(1.0 - c0 * c0 / 720.0,
                     -c0 / 6.0 * (1.0 - c1 / 20.0 * (1.0 - c1 / 42.0))));

  f1 = where(latboo_c1,
             (exp2iu * 2.0 * u -
              expmiu * complex(2.0 * u * cosw, (ww - 3.0 * uu) * xi0)) /
                 denom,
             complex(c0 / 24.0 * (1.0 - c1 / 15.0 * (1.0 - 3.0 * c1 / 112.0)),
                     1.0 - c1 / 6.0 * (1.0 - c1 / 20.0 * (1.0 - c1 / 42.0)) -
                         c0 * c0 / 5040.0));

  f2 = where(latboo_c1, (exp2iu - expmiu * complex(cosw, 3 * u * xi0)) / denom,
             complex(-1.0 + c1 / 12.0 * (1 - c1 / 30.0 * (1.0 - c1 / 56.0)) +
                         c0 * c0 / 20160.0,
                     c0 / 60.0 * (1.0 - c1 / 21.0 * (1.0 - c1 / 48.0))) *
                 0.5);

  // f_j(-c0, c1) = (-1)^j f*_j(c0, c1)
  f0 = where(latboo_c0 && latboo_c1, f0.conj(), f0);
  f1 = where(latboo_c0 && latboo_c1, -f1.conj(), f1);
  f2 = where(latboo_c0 && latboo_c1, f2.conj(), f2);

  // evaluate f0 + f1 Q + f2 QQ (= exp(iQ)) back into iQ
  iQ = QQ * f2 + Q * f1 + f0;

  // Relpace Q by exp(iQ)
  // Q = expiQ;
}
}

#endif
