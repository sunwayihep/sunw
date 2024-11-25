
#ifndef STAPLE_H
#define STAPLE_H

#include <complex.h>
#include <matrixsun.h>
#include <constants.h>
#include <index.h>
#include <device_load_save.h>

namespace CULQCD {
/**
  @brief Calculate the staple along direction mu and nu in even odd lattice
  array
  @param array gauge field
  @param staple store staple result
  @param idx 1D lattice index
  @param mu direction
  @param nu direction
  @param oddbit parity of the current lattice site.
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ void inline Staple(complex *array, msun &staple, int idx, int mu,
                              int nu, int oddbit) {
  msun link;
  int nuvolume = nu * DEVPARAMS::Volume;
  int muvolume = mu * DEVPARAMS::Volume;
  // UP
  link = GAUGE_LOAD<UseTex, atype, Real>(
      array, idx + oddbit * DEVPARAMS::HalfVolume + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_Neig_EO(idx, oddbit, nu, 1) + muvolume);
  link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
      array, Index_ND_Neig_EO(idx, oddbit, mu, 1) + nuvolume);
  staple += link;
  // DOWN
  link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
      array, Index_ND_Neig_EO(idx, oddbit, nu, -1) + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_Neig_EO(idx, oddbit, nu, -1) + muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_ND_Neig_EO(idx, oddbit, mu, 1, nu, -1) + nuvolume);
  staple += link;
}

/**
  @brief Calculate the staple along direction mu and nu in normal lattice array.
  @param array gauge field
  @param staple store staple result
  @param idx 1D lattice index
  @param mu direction
  @param nu direction
*/
template <bool UseTex, ArrayType atype, class Real>
__device__ void inline Staple(complex *array, msun &staple, int idx, int mu,
                              int nu) {
  msun link;
  int nuvolume = nu * DEVPARAMS::Volume;
  int muvolume = mu * DEVPARAMS::Volume;
  // UP
  link = GAUGE_LOAD<UseTex, atype, Real>(array, idx + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>(array, Index_4D_Neig_NM(idx, nu, 1) +
                                                     muvolume);
  link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
      array, Index_4D_Neig_NM(idx, mu, 1) + nuvolume);
  staple += link;
  // DOWN
  link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
      array, Index_4D_Neig_NM(idx, nu, -1) + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>(array, Index_4D_Neig_NM(idx, nu, -1) +
                                                     muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>(
      array, Index_4D_Neig_NM(idx, mu, 1, nu, -1) + nuvolume);
  staple += link;
}

/*template <bool evenoddarray, bool UseTex, ArrayType atype, class Real>
  __device__ void inline
  Staple(
  complex *array,
  msun &staple,
  int idx,
  int mu,
  int nu,
  int oddbit
  ){
  msun link;
  int nuvolume = nu * DEVPARAMS::Volume;
  int muvolume = mu * DEVPARAMS::Volume;
  if(evenoddarray){
  //UP
  link = GAUGE_LOAD<UseTex, atype, Real>( array,  idx + oddbit *
  DEVPARAMS::HalfVolume + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit,
  nu, 1) +muvolume );
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit,
  mu, 1) + nuvolume ).dagger();
  staple += link;
  //DOWN
  link = GAUGE_LOAD<UseTex, atype, Real>( array,  neighborEOIndex(idx, oddbit,
  nu, -1) +nuvolume ).dagger();
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit,
  nu, -1)  + muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborEOIndex(idx, oddbit,
  mu, 1, nu,  -1)+ nuvolume);
  staple += link;
  }
  else{
  //UP
  link = GAUGE_LOAD<UseTex, atype, Real>( array, idx + nuvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, nu, 1) +
  muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, mu, 1) +
  nuvolume).dagger();
  staple += link;
  //DOWN
  link = GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, nu, -1) +
  nuvolume).dagger();
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, nu, -1) +
  muvolume);
  link *= GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndex(idx, mu, 1, nu,
  -1) + nuvolume);
  staple += link;
  }

  }*/
}
#endif
