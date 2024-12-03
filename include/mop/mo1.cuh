#pragma once
#include <mop/wloop_arg.h>
namespace CULQCD {
//////////////////////////////////////////////////
//      _______                                 //
//     /      /         Alireza                 //
//    /ly    /          MO1                     //
//    |      |                                  //
//    |lx    |                                  //
//////////////////////////////////////////////////
// another version of MO1
template <bool UseTex, class Real, ArrayType atype>
DEVICE msun MO1(WLOPArg<Real> arg, int id, int lx, int ly, int muvolume) {
  msun mop = msun::zero();
  {
    int dir1 = arg.mu;
    int dmu[2], dmu1[2];
    dmu[0] = (dir1 + 1) % 3;
    dmu[1] = (dir1 + 2) % 3;
    dmu1[0] = dmu[1];
    dmu1[1] = dmu[0];
    int ids[2]; // these are just two variables update by moving on the path
    msun right, left;
    for (int il = 0; il < 2; il++) {
      right = msun::identity();
      left = msun::identity();
      ids[0] = id;
      ids[1] = Index_ND_Neig_NM(id, dir1, arg.radius);
      // 1st up
      for (int ix = 0; ix < lx; ix++) {
        left *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        right *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], 1);
      }
      // 2nd part
      {
        // 2nd forward
        msun link = msun::identity();
        for (int iy = 0; iy < ly; iy++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], 1);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int iy = 0; iy < ly; iy++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], -1);
          link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        mop += left * link * right.dagger();
        // 2nd backward
        // refreshing variables
        link = msun::identity();
        ids[0] = Index_ND_Neig_NM(id, dmu[il], lx); // first(up)
        for (int iy = 0; iy < ly; iy++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], -1);
          link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int iy = 0; iy < ly; iy++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], 1);
        }
        mop += left * link * right.dagger();
      } // end of block for 2nd part(forward)
      // 1st part down
      // refreshing variables
      right = msun::identity();
      left = msun::identity();
      ids[0] = id;
      ids[1] = Index_ND_Neig_NM(id, dir1, arg.radius);
      // 1st down
      for (int ix = 0; ix < lx; ix++) {
        ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], -1);
        left *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        right *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      // 2nd part
      {
        // 2nd forward
        msun link = msun::identity();
        for (int iy = 0; iy < ly; iy++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], 1);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int iy = 0; iy < ly; iy++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], -1);
          link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        mop += left * link * right.dagger();
        // 2nd backward
        // refreshing variables
        link = msun::identity();
        ids[0] = Index_ND_Neig_NM(id, dmu[il], -lx); // 1st down
        for (int iy = 0; iy < ly; iy++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], -1);
          link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int iy = 0; iy < ly; iy++) {
          link *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], 1);
        }
        mop += left * link * right.dagger();
      } // end of block 2nd part(backward)
    } // end of for(il)
  } // end of block
  mop /= (2.0 * sqrt(2.0));
  return mop;
}

} // namespace CULQCD
