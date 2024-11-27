#pragma once
#include <mop/wloop_arg.h>
namespace CULQCD {
///////////////////////////////////////
//      _________                    //
//      |lx2    |  Alireza           //
//      /       /ly  N=16            //
//      |lx1    |                    //
//      MO2                          //
///////////////////////////////////////
template <bool UseTex, class Real, ArrayType atype>
DEVICE msun MO2(WLOPArg<Real> arg, int id, int lx1, int ly, int lx2,
                int muvolume) {
  msun mop = msun::zero();
  {
    int dir1 = arg.mu;
    int dmu[2], dmu1[2];
    dmu[0] = (dir1 + 1) % 3;
    dmu[1] = (dir1 + 2) % 3;
    dmu1[0] = dmu[1];
    dmu1[1] = dmu[0];
    int ids[2]; // ids[0] for the left side of the path, and ids[1] for right
                // side
    msun link0, right, left;
    for (int il = 0; il < 2; il++) {
      ids[0] = id;
      ids[1] = Index_ND_Neig_NM(id, dir1, arg.radius);
      left = msun::identity();
      right = msun::identity();
      // 1st up
      for (int ix1 = 0; ix1 < lx1; ix1++) {
        left *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        right *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], 1);
      }
      // 2nd forward
      {
        msun leftp = msun::identity();  // leftprim
        msun rightp = msun::identity(); // rightprim
        for (int iy = 0; iy < ly; iy++) {
          leftp *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], 1);
          rightp *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[1] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[1] = Index_ND_Neig_NM(ids[1], dmu1[il], 1);
        }
        // 3p up
        link0 = msun::identity();
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
        // 3p down
        // refreshing variables
        link0 = msun::identity();
        ids[0] = Index_ND_Neig_NM(id, dmu[il], lx1, dmu1[il],
                                  ly); // 1st up, 2nd forward
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
      }
      // end of 2nd part's block
      ///////////////////
      // 2nd backward
      // refreshing coordinates
      ids[0] = Index_ND_Neig_NM(id, dmu[il], lx1); // 1st up
      ids[1] = Index_ND_Neig_NM(id, dir1, arg.radius, dmu[il],
                                lx1); // right side coordinates
      {
        msun leftp = msun::identity();
        msun rightp = msun::identity();
        for (int iy = 0; iy < ly; iy++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], -1);
          leftp *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[1] = Index_ND_Neig_NM(ids[1], dmu1[il], -1);
          rightp *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[1] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        // 3p up
        link0 = msun::identity();
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
        // 3p down
        // refreshing variables
        link0 = msun::identity();
        ids[0] = Index_ND_Neig_NM(id, dmu[il], lx1, dmu1[il],
                                  -ly); // 2nd part is backward
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
      }
      // end of 2nd block
      //////////////////////////////////
      //      1 part downway          //
      //////////////////////////////////
      /*refreshing coordinates*/
      ids[0] = id;
      ids[1] = Index_ND_Neig_NM(id, dir1, arg.radius);
      left = msun::identity();
      right = msun::identity();
      for (int ix1 = 0; ix1 < lx1; ix1++) {
        ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
        left *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], -1);
        right *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      // second forward
      {
        msun leftp = msun::identity();
        msun rightp = msun::identity();
        for (int iy = 0; iy < ly; iy++) {
          leftp *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], 1);
          rightp *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[1] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[1] = Index_ND_Neig_NM(ids[1], dmu1[il], 1);
        }
        // 3p up
        link0 = msun::identity();
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
        // 3p down
        link0 = msun::identity();
        ids[0] = Index_ND_Neig_NM(id, dmu[il], -lx1, dmu1[il],
                                  ly); // 1st down, 2nd forward
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
      }
      // end of first block
      // refreshing coordinates
      ids[0] = Index_ND_Neig_NM(id, dmu[il], -lx1); // 1st down
      ids[1] = Index_ND_Neig_NM(id, dir1, arg.radius, dmu[il], -lx1);
      // 2nd backward
      {
        msun leftp = msun::identity();
        msun rightp = msun::identity();
        for (int iy = 0; iy < ly; iy++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu1[il], -1);
          ids[1] = Index_ND_Neig_NM(ids[1], dmu1[il], -1);
          leftp *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          rightp *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[1] + dmu1[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        // 3p up
        link0 = msun::identity();
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
        // 3p down
        link0 = msun::identity();
        ids[0] = Index_ND_Neig_NM(id, dmu[il], -lx1, dmu1[il],
                                  -ly); // 1st down, 2nd backward
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], -1);
          link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
        }
        for (int ir = 0; ir < arg.radius; ir++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
        }
        for (int ix2 = 0; ix2 < lx2; ix2++) {
          link0 *= GAUGE_LOAD<UseTex, atype, Real>(
              arg.gaugefield, ids[0] + dmu[il] * DEVPARAMS::Volume,
              DEVPARAMS::size);
          ids[0] = Index_ND_Neig_NM(ids[0], dmu[il], 1);
        }
        mop += left * leftp * link0 * (right * rightp).dagger();
      }
    } // end of for(il)
  } // end of the block, all other variable are inside this block expect mop
  mop *= 0.25;
  return mop;
}

} // namespace CULQCD
