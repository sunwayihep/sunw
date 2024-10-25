

#ifndef DEVICE_PHB_OVR_H
#define DEVICE_PHB_OVR_H

#include <random.h>
#include <complex.h>
#include <matrixsun.h>
#include <msu2.h>
#include <constants.h>

namespace CULQCD {

/**
    @brief Generate full SU(2) matrix (four real numbers instead of 2x2 complex
   matrix) and update link matrix
    @param al weight
    @param localstate CURAND rng state
*/
template <class Real>
__device__ inline msu2 generate_su2_matrix(Real ap, cuRNGState &localState);

/**
    @brief Generate full SU(2) matrix (four real numbers instead of 2x2 complex
   matrix) and update link matrix.
    Get from MILC code.
    @param al weight
    @param localstate CURAND rng state
*/
template <class Real>
__device__ inline msu2 generate_su2_matrix_milc(Real al,
                                                cuRNGState &localState);

/**
    @brief Return SU(2) subgroup (4 real numbers) from SU(3) matrix
    @param tmp1 input SU(3) matrix
    @param block to retrieve from 0 to 2.
    @return 4 real numbers
*/
template <class Real>
__host__ __device__ inline msu2 get_block_su2(msu3 tmp1, int block);

/**
    @brief Return SU(2) subgroup (4 real numbers) from SU(Nc) matrix
    @param tmp1 input SU(Nc) matrix
    @param id the two indices to retrieve SU(2) block
    @return 4 real numbers
*/
template <class Real>
__host__ __device__ inline msu2 get_block_su2(msun tmp1, int2 id);

/**
    @brief Create a SU(Nc) identity matrix and fills with the SU(2) block
    @param rr SU(2) matrix represented only by four real numbers
    @param id the two indices to fill in the SU(3) matrix
    @return SU(Nc) matrix
*/
template <class Real>
__host__ __device__ inline msun block_su2_to_sun(msu2 rr, int2 id);

/**
    @brief Update the SU(Nc) link with the new SU(2) matrix, link <- u * link
    @param u SU(2) matrix represented by four real numbers
    @param link SU(Nc) matrix
    @param id indices
*/
template <class Real>
__host__ __device__ inline void mul_block_sun(msu2 u, msun &link, int2 id);

/**
    @brief Update the SU(3) link with the new SU(2) matrix, link <- u * link
    @param U SU(3) matrix
    @param a00 element (0,0) of the SU(2) matrix
    @param a01 element (0,1) of the SU(2) matrix
    @param a10 element (1,0) of the SU(2) matrix
    @param a11 element (1,1) of the SU(2) matrix
    @param block of the SU(3) matrix, 0,1 or 2
*/
template <class Real>
__host__ __device__ inline void block_su2_to_su3(msu3 &U, complex a00,
                                                 complex a01, complex a10,
                                                 complex a11, int block);

/**
    @brief Link update by pseudo-heatbath
    @param U link to be updated
    @param F staple
    @param localstate CURAND rng state
*/
template <class Real>
__device__ inline void heatBathSUN(msun &U, msun F, cuRNGState &localState);

/**
    @brief Link update by overrelaxation
    @param U link to be updated
    @param F staple
*/
template <class Real> __device__ inline void overrelaxationSUN(msun &U, msun F);

/**
    @brief Generate the four random real elements of the SU(2) matrix
    @param localstate CURAND rng state
    @return four real numbers of the SU(2) matrix
*/
template <class Real> __device__ inline msu2 randomSU2(cuRNGState &localState);

/**
    @brief Generate a SU(Nc) random matrix
    @param localstate CURAND rng state
    @return SU(Nc) matrix
*/
template <class Real> __device__ inline msun randomize(cuRNGState &localState);

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    @brief Generate full SU(2) matrix (four real numbers instead of 2x2 complex
   matrix) and update link matrix
    @param al weight
    @param localstate CURAND rng state
*/
template <class Real>
__device__ inline msu2 generate_su2_matrix(Real ap, cuRNGState &localState) {
  Real x, delta;
  msu2 a;
  do {
    Real X1 = -log(Random<Real>(localState)) / ap;

    Real X2 = -log(Random<Real>(localState)) / ap;
    Real C = pow2(cos(2.0 * PII * Random<Real>(localState)));
    delta = X1 * C + X2;
    x = Random<Real>(localState);
  } while (x * x > 1.0 - 0.5 * delta);
  a.a0() = 1.0 - delta;
  Real aabs = sqrt(1.0 - a.a0() * a.a0()),
       cos_theta = Random<Real>(&localState, -1.0, 1.0),
       phi = PII * Random<Real>(localState);
  Real sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  a.a1() = aabs * cos_theta;
  a.a2() = aabs * sin_theta * cos(phi);
  a.a3() = aabs * sin_theta * sin(phi);
  return a;
}

/**
    @brief Generate full SU(2) matrix (four real numbers instead of 2x2 complex
   matrix) and update link matrix.
    Get from MILC code.
    @param al weight
    @param localstate CURAND rng state
*/
template <class Real>
__device__ inline msu2 generate_su2_matrix_milc(Real al,
                                                cuRNGState &localState) {
  Real xr1, xr2, xr3, xr4, d, r;
  int k;
  xr1 = Random<Real>(localState);
  xr1 = (log((xr1 + 1.e-10)));
  xr2 = Random<Real>(localState);
  xr2 = (log((xr2 + 1.e-10)));
  xr3 = Random<Real>(localState);
  xr4 = Random<Real>(localState);
  xr3 = cos(PII * xr3);
  d = -(xr2 + xr1 * xr3 * xr3) / al;
  // now  beat each  site into submission
  int nacd = 0;
  if ((1.00 - 0.5 * d) > xr4 * xr4)
    nacd = 1;
  if (nacd == 0 && al > 2.0) { // k-p algorithm
    for (k = 0; k < 20; k++) {
      // get four random numbers (add a small increment to prevent taking
      // log(0.)
      xr1 = Random<Real>(localState);
      xr1 = (log((xr1 + 1.e-10)));
      xr2 = Random<Real>(localState);
      xr2 = (log((xr2 + 1.e-10)));
      xr3 = Random<Real>(localState);
      xr4 = Random<Real>(localState);
      xr3 = cos(PII * xr3);
      d = -(xr2 + xr1 * xr3 * xr3) / al;
      if ((1.00 - 0.5 * d) > xr4 * xr4)
        break;
    }
  } // endif nacd
  msu2 a;
  if (nacd == 0 && al <= 2.0) { // creutz algorithm
    xr3 = exp(-2.0 * al);
    xr4 = 1.0 - xr3;
    for (k = 0; k < 20; k++) {
      // get two random numbers
      xr1 = Random<Real>(localState);
      xr2 = Random<Real>(localState);
      r = xr3 + xr4 * xr1;
      a.a0() = 1.00 + log(r) / al;
      if ((1.0 - a.a0() * a.a0()) > xr2 * xr2)
        break;
    }
    d = 1.0 - a.a0();
  } // endif nacd
  // generate the four su(2) elements
  // find a0  = 1 - d
  a.a0() = 1.0 - d;
  // compute r
  xr3 = 1.0 - a.a0() * a.a0();
  xr3 = abs(xr3);
  r = sqrt(xr3);
  // compute a3
  a.a3() = (2.0 * Random<Real>(localState) - 1.0) * r;
  // compute a1 and a2
  xr1 = xr3 - a.a3() * a.a3();
  xr1 = abs(xr1);
  xr1 = sqrt(xr1);
  // xr2 is a random number between 0 and 2*pi
  xr2 = PII * Random<Real>(localState);
  a.a1() = xr1 * cos(xr2);
  a.a2() = xr1 * sin(xr2);
  return a;
}

/**
    @brief Return SU(2) subgroup (4 real numbers) from SU(3) matrix
    @param tmp1 input SU(3) matrix
    @param block to retrieve from 0 to 2.
    @return 4 real numbers
*/
template <class Real>
__host__ __device__ inline msu2 get_block_su2(msu3 tmp1, int block) {
  msu2 r;
  switch (block) {
  case 0:
    r.a0() = tmp1.e[0][0].real() + tmp1.e[1][1].real();
    r.a1() = tmp1.e[0][1].imag() + tmp1.e[1][0].imag();
    r.a2() = tmp1.e[0][1].real() - tmp1.e[1][0].real();
    r.a3() = tmp1.e[0][0].imag() - tmp1.e[1][1].imag();
    break;
  case 1:
    r.a0() = tmp1.e[1][1].real() + tmp1.e[2][2].real();
    r.a1() = tmp1.e[1][2].imag() + tmp1.e[2][1].imag();
    r.a2() = tmp1.e[1][2].real() - tmp1.e[2][1].real();
    r.a3() = tmp1.e[1][1].imag() - tmp1.e[2][2].imag();
    break;
  case 2:
    r.a0() = tmp1.e[0][0].real() + tmp1.e[2][2].real();
    r.a1() = tmp1.e[0][2].imag() + tmp1.e[2][0].imag();
    r.a2() = tmp1.e[0][2].real() - tmp1.e[2][0].real();
    r.a3() = tmp1.e[0][0].imag() - tmp1.e[2][2].imag();
    break;
  }
  return r;
}

/**
    @brief Return SU(2) subgroup (4 real numbers) from SU(Nc) matrix
    @param tmp1 input SU(Nc) matrix
    @param id the two indices to retrieve SU(2) block
    @return 4 real numbers
*/
template <class Real>
__host__ __device__ inline msu2 get_block_su2(msun tmp1, int2 id) {
  msu2 r;
  r.a0() = tmp1.e[id.x][id.x].real() + tmp1.e[id.y][id.y].real();
  r.a1() = tmp1.e[id.x][id.y].imag() + tmp1.e[id.y][id.x].imag();
  r.a2() = tmp1.e[id.x][id.y].real() - tmp1.e[id.y][id.x].real();
  r.a3() = tmp1.e[id.x][id.x].imag() - tmp1.e[id.y][id.y].imag();
  return r;
}

/**
    @brief Create a SU(Nc) identity matrix and fills with the SU(2) block
    @param rr SU(2) matrix represented only by four real numbers
    @param id the two indices to fill in the SU(3) matrix
    @return SU(Nc) matrix
*/
template <class Real>
__host__ __device__ inline msun block_su2_to_sun(msu2 rr, int2 id) {
  msun tmp1 = msun::unit();
  tmp1.e[id.x][id.x] = complex(rr.a0(), rr.a3());
  tmp1.e[id.x][id.y] = complex(rr.a2(), rr.a1());
  tmp1.e[id.y][id.x] = complex(-rr.a2(), rr.a1());
  tmp1.e[id.y][id.y] = complex(rr.a0(), -rr.a3());
  return tmp1;
}
/**
    @brief Update the SU(Nc) link with the new SU(2) matrix, link <- u * link
    @param u SU(2) matrix represented by four real numbers
    @param link SU(Nc) matrix
    @param id indices
*/
template <class Real>
__host__ __device__ inline void mul_block_sun(msu2 u, msun &link, int2 id) {
  complex tmp;
  for (int j = 0; j < NCOLORS; j++) {
    tmp = complex(u.a0(), u.a3()) * link.e[id.x][j] +
          complex(u.a2(), u.a1()) * link.e[id.y][j];
    link.e[id.y][j] = complex(-u.a2(), u.a1()) * link.e[id.x][j] +
                      complex(u.a0(), -u.a3()) * link.e[id.y][j];
    link.e[id.x][j] = tmp;
  }
}

/**
    @brief Update the SU(3) link with the new SU(2) matrix, link <- u * link
    @param U SU(3) matrix
    @param a00 element (0,0) of the SU(2) matrix
    @param a01 element (0,1) of the SU(2) matrix
    @param a10 element (1,0) of the SU(2) matrix
    @param a11 element (1,1) of the SU(2) matrix
    @param block of the SU(3) matrix, 0,1 or 2
*/
template <class Real>
__host__ __device__ inline void block_su2_to_su3(msu3 &U, complex a00,
                                                 complex a01, complex a10,
                                                 complex a11, int block) {
  complex tmp;
  switch (block) {
  case 0:
    tmp = a00 * U.e[0][0] + a01 * U.e[1][0];
    U.e[1][0] = a10 * U.e[0][0] + a11 * U.e[1][0];
    U.e[0][0] = tmp;
    tmp = a00 * U.e[0][1] + a01 * U.e[1][1];
    U.e[1][1] = a10 * U.e[0][1] + a11 * U.e[1][1];
    U.e[0][1] = tmp;
    tmp = a00 * U.e[0][2] + a01 * U.e[1][2];
    U.e[1][2] = a10 * U.e[0][2] + a11 * U.e[1][2];
    U.e[0][2] = tmp;
    break;
  case 1:
    tmp = a00 * U.e[1][0] + a01 * U.e[2][0];
    U.e[2][0] = a10 * U.e[1][0] + a11 * U.e[2][0];
    U.e[1][0] = tmp;
    tmp = a00 * U.e[1][1] + a01 * U.e[2][1];
    U.e[2][1] = a10 * U.e[1][1] + a11 * U.e[2][1];
    U.e[1][1] = tmp;
    tmp = a00 * U.e[1][2] + a01 * U.e[2][2];
    U.e[2][2] = a10 * U.e[1][2] + a11 * U.e[2][2];
    U.e[1][2] = tmp;
    break;
  case 2:
    tmp = a00 * U.e[0][0] + a01 * U.e[2][0];
    U.e[2][0] = a10 * U.e[0][0] + a11 * U.e[2][0];
    U.e[0][0] = tmp;
    tmp = a00 * U.e[0][1] + a01 * U.e[2][1];
    U.e[2][1] = a10 * U.e[0][1] + a11 * U.e[2][1];
    U.e[0][1] = tmp;
    tmp = a00 * U.e[0][2] + a01 * U.e[2][2];
    U.e[2][2] = a10 * U.e[0][2] + a11 * U.e[2][2];
    U.e[0][2] = tmp;
    break;
  }
}

/**
    @brief Link update by pseudo-heatbath
    @param U link to be updated
    @param F staple
    @param localstate CURAND rng state
*/
template <class Real>
__device__ inline void heatBathSUN(msun &U, msun F, cuRNGState &localState) {
#if (NCOLORS == 3)
  //////////////////////////////////////////////////////////////////
  /*
    for( int block = 0; block < NCOLORS; block++ ) {
    msu3 tmp1 = U * F;
    msu2 r = get_block_su2<Real>(tmp1, block);
    Real k = r.abs();
    Real ap = (Real)DEVPARAMS::BetaOverNc * k;
    k = (Real)1.0 / k;
    r *= k;
    //msu2 a = generate_su2_matrix<T4, T>(ap, localState);
    msu2 a = generate_su2_matrix_milc<Real>(ap, localState);
    r = mulsu2UVDagger_4<Real>( a, r);
    ///////////////////////////////////////
    block_su2_to_su3<Real>( U, complex( r.a0(), r.a3() ), complex( r.a2(),
    r.a1() ), complex(-r.a2(), r.a1() ), complex( r.a0(),-r.a3() ), block );
    //FLOP_min = (198 + 4 + 15 + 28 + 28 + 84) * 3 = 1071
    }*/
  //////////////////////////////////////////////////////////////////

  for (int block = 0; block < NCOLORS; block++) {
    int p, q;
    IndexBlock(block, p, q);
    complex a0 = complex::zero();
    complex a1 = complex::zero();
    complex a2 = complex::zero();
    complex a3 = complex::zero();

    for (int j = 0; j < NCOLORS; j++) {
      a0 += U.e[p][j] * F.e[j][p];
      a1 += U.e[p][j] * F.e[j][q];
      a2 += U.e[q][j] * F.e[j][p];
      a3 += U.e[q][j] * F.e[j][q];
    }
    msu2 r;
    r.a0() = a0.real() + a3.real();
    r.a1() = a1.imag() + a2.imag();
    r.a2() = a1.real() - a2.real();
    r.a3() = a0.imag() - a3.imag();
    Real k = r.abs();
    Real ap = (Real)DEVPARAMS::BetaOverNc * k;
    k = (Real)1.0 / k;
    r *= k;
    // msu2 a = generate_su2_matrix<T4, T>(ap, localState);
    msu2 a = generate_su2_matrix_milc<Real>(ap, localState);
    r = mulsu2UVDagger<Real>(a, r);
    ///////////////////////////////////////
    a0 = complex(r.a0(), r.a3());
    a1 = complex(r.a2(), r.a1());
    a2 = complex(-r.a2(), r.a1());
    a3 = complex(r.a0(), -r.a3());
    complex tmp0;

    for (int j = 0; j < NCOLORS; j++) {
      tmp0 = a0 * U.e[p][j] + a1 * U.e[q][j];
      U.e[q][j] = a2 * U.e[p][j] + a3 * U.e[q][j];
      U.e[p][j] = tmp0;
    }
    // FLOP_min = (NCOLORS * 64 + 19 + 28 + 28) * 3 = NCOLORS * 192 + 225
  }
//////////////////////////////////////////////////////////////////

#elif(NCOLORS == 2)
  int2 id = IndexBlock(0);
  msu2 r = get_block_su2<Real>(F, id);
  Real k = r.abs();

  /**
  if (INDEX1D() == 0) {
    printf("print U matrix\n");
    U.print();

    printf("print F matrix\n");
    F.print();

    printf("print r matrix\n");
    r.print();

    printf("k = %f\n", k);
  }*/

  Real ap = (Real)DEVPARAMS::BetaOverNc * k;
  k = (Real)1.0 / k;
  r *= k;
  // msu2 a = generate_su2_matrix<T4, T>(ap, localState);
  msu2 a = generate_su2_matrix_milc<Real>(ap, localState);
  msu2 rr = mulsu2UVDagger<Real>(a, r);

  int2 id2;
  id2.x = 0, id2.y = 1;
  U = block_su2_to_sun(rr, id2);
/**
if (INDEX1D() == 0) {
  printf("print U matrix\n");
  U.print();
}*/

#else
  //////////////////////////////////////////////////////////////////
  // TESTED IN SU(4) SP THIS IS WORST
  msun M = U * F;
  for (int block = 0; block < NCOLORS * (NCOLORS - 1) / 2; block++) {
    int2 id = IndexBlock(block);
    msu2 r = get_block_su2<Real>(M, id);
    Real k = r.abs();
    Real ap = (Real)DEVPARAMS::BetaOverNc * k;
    k = (Real)1.0 / k;
    r *= k;
    // msu2 a = generate_su2_matrix<T4, T>(ap, localState);
    msu2 a = generate_su2_matrix_milc<Real>(ap, localState);
    msu2 rr = mulsu2UVDagger<Real>(a, r);
    ///////////////////////////////////////
    mul_block_sun<Real>(rr, U, id);
    mul_block_sun<Real>(rr, M, id);
    ///////////////////////////////////////
  }
/*//TESTED IN SU(4) SP THIS IS FASTER
for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
    int2 id = IndexBlock( block );
    complex a0 = complex::zero();
    complex a1 = complex::zero();
    complex a2 = complex::zero();
    complex a3 = complex::zero();

    for(int j = 0; j < NCOLORS; j++){
        a0 += U.e[id.x][j] * F.e[j][id.x];
        a1 += U.e[id.x][j] * F.e[j][id.y];
        a2 += U.e[id.y][j] * F.e[j][id.x];
        a3 += U.e[id.y][j] * F.e[j][id.y];
    }
    msu2 r;
    r.a0() = a0.real() + a3.real();
    r.a1() = a1.imag() + a2.imag();
    r.a2() = a1.real() - a2.real();
    r.a3() = a0.imag() - a3.imag();
    Real k = r.abs();
    Real ap = (Real)DEVPARAMS::BetaOverNc * k;
    k = (Real)1.0 / k;
    r *= k;
    //msu2 a = generate_su2_matrix<T4, T>(ap, localState);
    msu2 a = generate_su2_matrix_milc<Real>(ap, localState);
    r = mulsu2UVDagger<Real>( a, r);
    mul_block_sun<Real>( r, U, id);*/
/*///////////////////////////////////////
  a0 = complex( r.a0(), r.a3() );
  a1 = complex( r.a2(), r.a1() );
  a2 = complex(-r.a2(), r.a1() );
  a3 = complex( r.a0(),-r.a3() );
  complex tmp0;

  for(int j = 0; j < NCOLORS; j++){
  tmp0 = a0 * U.e[id.x][j] + a1 * U.e[id.y][j];
  U.e[id.y][j] = a2 * U.e[id.x][j] + a3 * U.e[id.y][j];
  U.e[id.x][j] = tmp0;
  }	*/
// }

#endif
  //////////////////////////////////////////////////////////////////
}

//////////////////////////////////////////////////////////////////////////
/**
    @brief Link update by overrelaxation
    @param U link to be updated
    @param F staple
*/
template <class Real>
__device__ inline void overrelaxationSUN(msun &U, msun F) {

#if (NCOLORS == 3)
  //////////////////////////////////////////////////////////////////
  /*
    for( int block = 0; block < 3; block++ ) {
    msu3 tmp1 = U * F;
    msu2 r = get_block_su2<Real>(tmp1, block);
    //normalize and conjugate
    r = r.conj_normalize();
    ///////////////////////////////////////
    complex a00 = complex( r.a0(), r.a3() );
    complex a01 = complex( r.a2(), r.a1() );
    complex a10 = complex(-r.a2(), r.a1() );
    complex a11 = complex( r.a0(),-r.a3() );
    block_su2_to_su3<Real>( U, a00, a01, a10, a11, block );
    block_su2_to_su3<Real>( U, a00, a01, a10, a11, block );

    //FLOP = (198 + 17 + 84 * 2) * 3 = 1149
    }*/
  ///////////////////////////////////////////////////////////////////
  // This version does not need to multiply all matrix at each block: tmp1 = U *
  // F;
  //////////////////////////////////////////////////////////////////

  for (int block = 0; block < 3; block++) {
    int p, q;
    IndexBlock(block, p, q);
    complex a0 = complex::zero();
    complex a1 = complex::zero();
    complex a2 = complex::zero();
    complex a3 = complex::zero();

    for (int j = 0; j < NCOLORS; j++) {
      a0 += U.e[p][j] * F.e[j][p];
      a1 += U.e[p][j] * F.e[j][q];
      a2 += U.e[q][j] * F.e[j][p];
      a3 += U.e[q][j] * F.e[j][q];
    }
    msu2 r;
    r.a0() = a0.real() + a3.real();
    r.a1() = a1.imag() + a2.imag();
    r.a2() = a1.real() - a2.real();
    r.a3() = a0.imag() - a3.imag();
    // normalize and conjugate
    r = r.conj_normalize();
    ///////////////////////////////////////
    a0 = complex(r.a0(), r.a3());
    a1 = complex(r.a2(), r.a1());
    a2 = complex(-r.a2(), r.a1());
    a3 = complex(r.a0(), -r.a3());
    complex tmp0, tmp1;

    for (int j = 0; j < NCOLORS; j++) {
      tmp0 = a0 * U.e[p][j] + a1 * U.e[q][j];
      tmp1 = a2 * U.e[p][j] + a3 * U.e[q][j];
      U.e[p][j] = a0 * tmp0 + a1 * tmp1;
      U.e[q][j] = a2 * tmp0 + a3 * tmp1;
    }
    // FLOP = (NCOLORS * 88 + 17) * 3
  }
///////////////////////////////////////////////////////////////////
#else
  ///////////////////////////////////////////////////////////////////
  msun M = U * F;
  for (int block = 0; block < NCOLORS * (NCOLORS - 1) / 2; block++) {
    int2 id = IndexBlock(block);
    msu2 r = get_block_su2<Real>(M, id);
    // normalize and conjugate
    r = r.conj_normalize();
    mul_block_sun<Real>(r, U, id);
    mul_block_sun<Real>(r, U, id);
    mul_block_sun<Real>(r, M, id);
    mul_block_sun<Real>(r, M, id);
    ///////////////////////////////////////
  }
/*	//TESTED IN SU(4) SP THIS IS WORST
    for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
    int2 id = IndexBlock( block );
    complex a0 = complex::zero();
    complex a1 = complex::zero();
    complex a2 = complex::zero();
    complex a3 = complex::zero();

    for(int j = 0; j < NCOLORS; j++){
            a0 += U.e[id.x][j] * F.e[j][id.x];
            a1 += U.e[id.x][j] * F.e[j][id.y];
            a2 += U.e[id.y][j] * F.e[j][id.x];
            a3 += U.e[id.y][j] * F.e[j][id.y];
    }
    msu2 r;
    r.a0() = a0.real() + a3.real();
    r.a1() = a1.imag() + a2.imag();
    r.a2() = a1.real() - a2.real();
    r.a3() = a0.imag() - a3.imag();
    //normalize and conjugate
    r = r.conj_normalize();
    //mul_block_sun<Real>( r, U, id);
    //mul_block_sun<Real>( r, U, id);
    ///////////////////////////////////////
    a0 = complex( r.a0(), r.a3() );
    a1 = complex( r.a2(), r.a1() );
    a2 = complex(-r.a2(), r.a1() );
    a3 = complex( r.a0(),-r.a3() );
    complex tmp0, tmp1;

    for(int j = 0; j < NCOLORS; j++){
    tmp0 = a0 * U.e[id.x][j] + a1 * U.e[id.y][j];
    tmp1 = a2 * U.e[id.x][j] + a3 * U.e[id.y][j];
    U.e[id.x][j] = a0 * tmp0 + a1 * tmp1;
    U.e[id.y][j] = a2 * tmp0 + a3 * tmp1;
    }
    }
*/
#endif
}

/**
    @brief Generate the four random real elements of the SU(2) matrix
    @param localstate CURAND rng state
    @return four real numbers of the SU(2) matrix
*/
template <class Real> __device__ inline msu2 randomSU2(cuRNGState &localState) {
  msu2 a;
  Real aabs, ctheta, stheta, phi;
  a.a0() = Random<Real>(localState, (Real) - 1.0, (Real)1.0);
  aabs = sqrt((Real)1.0 - a.a0() * a.a0());
  ctheta = Random<Real>(localState, (Real) - 1.0, (Real)1.0);
  phi = PII * Random<Real>(localState);
  stheta = (Real)(curand(&localState) & 1 ? 1 : -1) *
           sqrt((Real)1.0 - ctheta * ctheta);
  a.a1() = aabs * stheta * cos(phi);
  a.a2() = aabs * stheta * sin(phi);
  a.a3() = aabs * ctheta;
  return a;
}

/**
    @brief Generate a SU(Nc) random matrix
    @param localstate CURAND rng state
    @return SU(Nc) matrix
*/
template <class Real> __device__ inline msun randomize(cuRNGState &localState) {
  msun U;

  for (int i = 0; i < NCOLORS; i++)
    for (int j = 0; j < NCOLORS; j++)
      U.e[i][j] = complex((Real)(Random<Real>(localState) - 0.5),
                          (Real)(Random<Real>(localState) - 0.5));
  /*msun U = msu3::unit();
    for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
    msu2 rr = randomSU2<Real>(localState);
    ///////////////////////////////////////
    complex a00 = complex::make_complex( rr.a0(), rr.a3() );
    complex a01 = complex::make_complex( rr.a2(), rr.a1() );
    complex a10 = complex::make_complex(-rr.a2(), rr.a1() );
    complex a11 = complex::make_complex( rr.a0(),-rr.a3() );
    U = block_su2_to_su3<Real>( U, a00, a01, a10, a11, block );
    }*/
  return U;
}
}

#endif
