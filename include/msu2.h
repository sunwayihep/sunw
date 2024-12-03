
#ifndef MATRIXSU2_H
#define MATRIXSU2_H

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_common.h>
#include <cuda_vector_types.h>

namespace CULQCD {

/**
    @brief Holds SU(2) matrix with only four parameters.
*/
template <class Real> class _msu2 {

public:
  typedef typename MakeVector<Real, 4>::type Real4;
  Real4 val;

  M_HOSTDEVICE _msu2<Real>() { val.x = val.y = val.z = val.w = (Real)0.0; }

  M_HOSTDEVICE _msu2<Real> &operator=(const Real REF(a)) {
    val.x = a;
    val.y = a;
    val.z = a;
    val.w = a;
    return *this;
  };

  M_HOSTDEVICE _msu2<Real> &operator=(const Real ARRAYREF(a, 4)) {
    val.x = a[0];
    val.y = a[1];
    val.z = a[2];
    val.w = a[3];
    return *this;
  };

  // return references to the T4 components
  M_HOSTDEVICE Real &a0() { return val.x; };
  M_HOSTDEVICE Real &a1() { return val.y; };
  M_HOSTDEVICE Real &a2() { return val.z; };
  M_HOSTDEVICE Real &a3() { return val.w; };

  M_HOSTDEVICE _msu2<Real> operator=(const _msu2<Real> REF(b)) {
    val.x = b.val.x;
    val.y = b.val.y;
    val.z = b.val.z;
    val.w = b.val.w;
    return *this;
  }
  M_HOST bool operator==(const _msu2<Real> &A) {
    _msu2<Real> B = *this;
    return !memcmp(&A, &B, sizeof(_msu2<Real>));
  }
  M_HOST bool operator!=(const _msu2<Real> &A) {
    _msu2<Real> B = *this;
    return memcmp(&A, &B, sizeof(_msu2<Real>));
  }

  M_HOSTDEVICE _msu2<Real> operator+(const _msu2<Real> REF(b)) const {
    return make_msu2(val.x + b.val.x, val.y + b.val.y, val.z + b.val.z,
                     val.w + b.val.w);
  }
  M_HOSTDEVICE _msu2<Real> operator+=(const _msu2<Real> REF(b)) {
    return *this = *this + b;
  }

  M_HOSTDEVICE _msu2<Real> operator+(const Real REF(b)) const {
    return make_msu2(val.x + b, val.y + b, val.z + b, val.w + b);
  }

  M_HOSTDEVICE _msu2<Real> operator-(const _msu2<Real> REF(b)) const {
    return make_msu2(val.x - b.val.x, val.y - b.val.y, val.z - b.val.z,
                     val.w - b.val.w);
  }
  M_HOSTDEVICE _msu2<Real> operator-=(const _msu2<Real> REF(b)) {
    return *this = *this - b;
  }

  M_HOSTDEVICE _msu2<Real> operator-() const {
    return make_msu2(-val.x, -val.y, -val.z, -val.w);
  }

  M_HOSTDEVICE _msu2<Real> operator-(const Real REF(b)) const {
    return make_msu2(val.x - b, val.y - b, val.z - b, val.w - b);
    ;
  }

  M_HOSTDEVICE _msu2<Real> operator*(const _msu2<Real> REF(v)) const {
    _msu2<Real> result;
    result.val.x =
        val.x * v.val.x - val.y * v.val.y - val.z * v.val.z - val.w * v.val.w;
    result.val.y =
        val.x * v.val.y + val.y * v.val.x - val.z * v.val.w + val.w * v.val.z;
    result.val.z =
        val.x * v.val.z + val.y * v.val.w + val.z * v.val.x - val.w * v.val.y;
    result.val.w =
        val.x * v.val.w - val.y * v.val.z + val.z * v.val.y + val.w * v.val.x;
    return result;
  }

  M_HOSTDEVICE _msu2<Real> operator*=(const _msu2<Real> REF(v)) {
    return *this = *this * v;
  }

  M_HOSTDEVICE _msu2<Real> operator*(const Real REF(b)) const {
    return make_msu2(val.x * b, val.y * b, val.z * b, val.w * b);
  }
  M_HOSTDEVICE _msu2<Real> operator*=(const Real REF(b)) {
    val.x *= b;
    val.y *= b;
    val.z *= b;
    val.w *= b;
    return *this;
  }

  M_HOSTDEVICE _msu2<Real> operator/(const Real REF(b)) const {
    return make_msu2(val.x / b, val.y / b, val.z / b, val.w / b);
  }

  M_HOSTDEVICE _msu2<Real> operator/=(const Real REF(b)) {
    return *this = *this / b;
  }

  // complex conjugate
  M_HOSTDEVICE _msu2<Real> operator~() const {
    return make_msu2(val.x, -val.y, -val.z, -val.w);
  }

  M_HOSTDEVICE _msu2<Real> conj() {
    return make_msu2(val.x, -val.y, -val.z, -val.w);
  }

  M_HOSTDEVICE Real det() {
    Real res = val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    return res;
  }

  M_HOSTDEVICE _msu2<Real> normalize() {
    Real norm = (Real)1.0 / sqrt(val.x * val.x + val.y * val.y + val.z * val.z +
                                 val.w * val.w);
    val.x *= norm;
    val.y *= norm;
    val.z *= norm;
    val.w *= norm;
    return *this;
  }
  M_HOSTDEVICE Real abs() {
    return sqrt(val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
  }
  M_HOSTDEVICE Real abs2() {
    return (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
  }

  M_HOSTDEVICE _msu2<Real> conj_normalize() {
    Real norm = (Real)1.0 / sqrt(val.x * val.x + val.y * val.y + val.z * val.z +
                                 val.w * val.w);
    val.x *= norm;
    val.y *= -norm;
    val.z *= -norm;
    val.w *= -norm;
    return *this;
  }

  M_HOSTDEVICE Real trace() {
    Real result = (Real)2.0 * val.x;
    return result;
  }

  static M_HOSTDEVICE _msu2<Real> make_msu2(Real a, Real b, Real c, Real d) {
    _msu2<Real> res;
    res.a0() = a;
    res.a1() = b;
    res.a2() = c;
    res.a3() = d;
    return res;
  }
  /*  static M_HOSTDEVICE _msu2<Real> make_msu2(T4 a){
      _msu2<Real> res;
      res.a0() = a.val.x;
      res.a1() = a.val.y;
      res.a2() = a.val.z;
      res.a3() = a.val.w;
      return res;
      }*/

  static M_HOSTDEVICE _msu2<Real> identity() {
    return make_msu2((Real)1.0, (Real)0.0, (Real)0.0, (Real)0.0);
  }

  // return constant number zero
  static M_HOSTDEVICE _msu2<Real> zero() {
    return make_msu2((Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0);
  }

  // print matrix contents (host code only!)
  M_HOSTDEVICE void print() {
    printf("%.10e\t%.10e\t%.10e\t%.10e\n", val.x, val.y, val.z, val.w);
  }
  M_HOST void print(FILE *stream) {
    fprintf(stream, "%.12e\t%.12e\t%.12e\t%.12e\n", val.x, val.y, val.z, val.w);
  }
  friend M_HOST std::ostream &operator<<(std::ostream &out, _msu2<Real> &M) {
    // out << std::setprecision(14);
    out << M.val.x << '\t' << M.val.y << '\t' << M.val.z << '\t' << M.val.w;
    return out;
  }
};

#define msu2 _msu2<Real>
typedef _msu2<float> msu2s;
typedef _msu2<double> msu2d;

// v * u^dagger
template <class Real>
__host__ __device__ inline msu2 mulsu2UVDagger(msu2 v, msu2 u) {
  msu2 b;
  b.a0() =
      v.a0() * u.a0() + v.a1() * u.a1() + v.a2() * u.a2() + v.a3() * u.a3();
  b.a1() =
      v.a1() * u.a0() - v.a0() * u.a1() + v.a2() * u.a3() - v.a3() * u.a2();
  b.a2() =
      v.a2() * u.a0() - v.a0() * u.a2() + v.a3() * u.a1() - v.a1() * u.a3();
  b.a3() =
      v.a3() * u.a0() - v.a0() * u.a3() + v.a1() * u.a2() - v.a2() * u.a1();
  return b;
}
} // namespace CULQCD

#endif // #ifndef MATRIXSU2_H
