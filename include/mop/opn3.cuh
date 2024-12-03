#pragma once
#include <mop/wloop_arg.h>
namespace CULQCD {

////////////////////////////////////
////////////////////////////////////
template <bool UseTex, class Real, ArrayType atype>
DEVICE msun N3(WLOPArg<Real> arg, int id, int l1, int muvolume, msun line_left,
               msun line_right) {
  msun mop = msun::zero();
  {
    ///////////////////////////////////////////////////
    //                                               //
    //  CALCULATE PATHS OF THE TYPE:                 //
    //                                               //
    //  _____                                        //
    //  |___|____ line                               //
    //  svec[i]                                      //
    ///////////////////////////////////////////////////
    int dir1 = arg.mu;
    int dir2 = (dir1 + 1) % 3;
    int dir3 = (dir1 + 2) % 3;
    int dmu[2];
    dmu[0] = dir2;
    dmu[1] = dir3;
    int halfline = arg.radius / 2;
    int ids[2];
    ids[0] = id;
    msun line0 = msun::identity();
    for (int ir = 0; ir < halfline; ++ir) {
      line0 *= GAUGE_LOAD<UseTex, atype, Real>(
          arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
      ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
    }
    // 2 comp, in upway
    for (int il = 0; il < 2; il++) {
      msun link0 = msun::identity();
      ids[1] = id;
      for (int i1 = 0; i1 < l1; i1++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], 1);
      }
      for (int ir = 0; ir < halfline; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + muvolume, DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dir1, 1);
      }
      for (int i1 = 0; i1 < l1; i1++) {
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      msun link = line0 * link0.dagger() * line0 * line_right;
      mop += link;
    }

    // 2 comp, in downway
    for (int il = 0; il < 2; il++) {
      ids[1] = id;
      msun link0 = msun::identity();
      for (int i1 = 0; i1 < l1; i1++) {
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      for (int ir = 0; ir < halfline; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + muvolume, DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dir1, 1);
      }
      for (int i1 = 0; i1 < l1; i1++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], 1);
      }
      msun link = line0 * link0.dagger() * line0 * line_right;
      mop += link;
    }
    ///////////////////////////////////////////////////
    //                                               //
    //  CALCULATE PATHS OF THE TYPE:                 //
    //                                               //
    //      _____                                    //
    //  ____|___| svec[i]                            //
    //   line                                        //
    ///////////////////////////////////////////////////
    halfline = (arg.radius + 1) / 2;
    int s2 = Index_ND_Neig_NM(id, dir1, halfline);
    ids[0] = s2;
    line0 = msun::identity();
    for (int ir = halfline; ir < arg.radius; ir++) {
      line0 *= GAUGE_LOAD<UseTex, atype, Real>(
          arg.gaugefield, ids[0] + muvolume, DEVPARAMS::size);
      ids[0] = Index_ND_Neig_NM(ids[0], dir1, 1);
    }
    // 2 comp, in upway
    for (int il = 0; il < 2; il++) {
      msun link0 = msun::identity();
      ids[1] = s2;
      for (int i1 = 0; i1 < l1; i1++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], 1);
      }
      for (int ir = halfline; ir < arg.radius; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + muvolume, DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dir1, 1);
      }
      for (int i1 = 0; i1 < l1; i1++) {
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      msun link = line_left * line0 * link0.dagger() * line0;
      mop += link;
    }

    // 2 comp, in downway
    for (int il = 0; il < 2; il++) {
      ids[1] = s2;
      msun link0 = msun::identity();
      for (int i1 = 0; i1 < l1; i1++) {
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      for (int ir = halfline; ir < arg.radius; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + muvolume, DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dir1, 1);
      }
      for (int i1 = 0; i1 < l1; i1++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[1] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[1] = Index_ND_Neig_NM(ids[1], dmu[il], 1);
      }
      msun link = line_left * line0 * link0.dagger() * line0;
      mop += link;
    }
  }
  mop /= (2.0 * sqrt(2.0));
  return mop;
}

} // namespace CULQCD
