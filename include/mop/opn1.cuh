#pragma once
#include <mop/wloop_arg.h>
namespace CULQCD {
template <bool UseTex, class Real, ArrayType atype>
DEVICE msun N1(WLOPArg<Real> arg, int id, int lx, int muvolume, msun line_left,
               msun line_right) {
  msun mop = msun::zero();
  int dir1 = arg.mu;

  int dmu[2] = {(dir1 + 1) % 3, (dir1 + 2) % 3};
  ///////////////////////////////////////////////////
  //                                               //
  //  CALCULATE PATHS OF THE TYPE:                 //
  //                                               //
  //  _____                                        //
  //  |   |____ line                               //
  //  svec[i]                                      //
  ///////////////////////////////////////////////////
  {
    int ids[2] = {id, id};
    int halfline = arg.radius / 2;
    for (int il = 0; il < 2; il++) {
      msun link0 = msun::identity();
      for (int ix = 0; ix < lx; ix++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], 1);
      }
      for (int ir = 0; ir < halfline; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dir1, 1);
      }
      for (int ix = 0; ix < lx; ix++) {
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      mop += link0 * line_right;
    }
    // 2 comp, in downway
    ids[0] = id;
    ids[1] = id;
    for (int il = 0; il < 2; il++) {
      msun link0 = msun::identity();
      for (int ix = 0; ix < lx; ix++) {
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      for (int ir = 0; ir < halfline; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dir1, 1);
      }
      for (int ix = 0; ix < lx; ix++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], 1);
      }
      mop += link0 * line_right;
    }
  }
  ///////////////////////////////////////////////////
  //                                               //
  //  CALCULATE PATHS OF THE TYPE:                 //
  //                                               //
  //      _____                                    //
  //  ____|   | svec[i]                            //
  //   line                                        //
  ///////////////////////////////////////////////////
  {
    int halfline = (arg.radius + 1) / 2;
    int s = Index_ND_Neig_NM(id, dir1, halfline);
    int ids[2];
    // 2 comp, in upway

    for (int il = 0; il < 2; il++) {
      ids[0] = s;
      ids[1] = s;
      msun link0 = msun::identity();
      for (int ix = 0; ix < lx; ix++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], 1);
      }
      for (int ir = halfline; ir < arg.radius; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dir1, 1);
      }
      for (int ix = 0; ix < lx; ix++) {
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      mop += line_left * link0;
    }

    // 2 comp, in downway
    for (int il = 0; il < 2; il++) {
      ids[0] = s;
      ids[1] = s;
      msun link0 = msun::identity();
      for (int ix = 0; ix < lx; ix++) {
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
      }
      for (int ir = halfline; ir < arg.radius; ++ir) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dir1, 1);
      }
      for (int ix = 0; ix < lx; ix++) {
        link0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume,
            DEVPARAMS::size);
        ids[il] = Index_ND_Neig_NM(ids[il], dmu[il], 1);
      }
      mop += line_left * link0;
    }
  }

  mop /= (2.0 * sqrt(2.0));
  return mop;
}
} // namespace CULQCD
