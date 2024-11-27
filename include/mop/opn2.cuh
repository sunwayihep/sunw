#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
//Nuno operators with 
//////// 2nd nuno operator /////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun N2( WLOPArg<Real> arg,int id, int l1, int l2, int muvolume, msun line_left, msun line_right){
	msun mop=msun::zero();
	{
		  //for(int np = 0; np < Nbase; np++) mop[np]=msun::zero();
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
    //   ____                                        //
	//  /   /                                        //
	//  |   |____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int dmu[2]; int dmu1[2];
	dmu[0] = dir2;
	dmu[1] = dir3;
	dmu1[0] = dir3;
	dmu1[1] = dir2;
	int halfline = arg.radius/2;
	//2 comp, 1st upway 2nd foward  0/1
	for(int il = 0; il < 2; il++){
		int ids=id;
		msun link0=msun::identity();
		for(int i1=0;i1<l1;i1++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids, dmu[il],1);
		}
		for(int i2=0;i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dmu1[il],1);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2;i2++){
			ids= Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i1=0;i1<l1;i1++){
		ids=Index_ND_Neig_NM(ids, dmu[il],-1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		}
		mop += link0 * line_right;
	}
	//2 comp, 1st upway 2nd backward  2/3
	for(int il = 0; il < 2; il++){
		int ids=id;
		msun link0 =msun::identity();
		for(int i1=0;i1<l1;i1++){
		link0*= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids = Index_ND_Neig_NM(ids, dmu[il], 1);
		}
		for(int i2=0; i2<l2;i2++){
			ids=Index_ND_Neig_NM(ids, dmu1[il],-1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int ir = 0; ir < halfline; ++ir) {
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
		ids = Index_ND_Neig_NM(ids, dir1, 1);
	}
	for(int i2=0;i2<l2;i2++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids, dmu1[il],1);
		}
	for(int i1=0;i1<l1;i1++){
		ids=Index_ND_Neig_NM(ids, dmu[il],-1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		mop += link0 * line_right;
	}
	//2 comp, 1st downway 2nd foward 4/5
	for(int il = 0; il < 2; il++){
		int ids=id;
		msun link0=msun::identity();
		for(int i1=0; i1<l1;i1++){
			ids=Index_ND_Neig_NM(ids, dmu[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i2=0; i2<l2; i2++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids, dmu1[il],1);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0;i2<l2;i2++){
			ids= Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i1=0; i1<l1; i1++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids, dmu[il],1);
		}
		mop += link0 * line_right;
	}
	//2 comp, 1st downway 2nd backward 6/7
	for(int il = 0; il < 2; il++){
		int ids=id;
		msun link0=msun::identity();
		for(int i1=0;i1<l1; i1++){
			ids=Index_ND_Neig_NM(ids, dmu[il], -1);
			link0*= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i2=0; i2<l2;i2++){
			ids = Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0;i2<l2;i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dmu1[il], 1);
		}
		for(int i1=0; i1<l1; i1++){
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids, dmu[il],1);
		}
		mop += link0 * line_right;
		}

	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//       ____                                    //
	//      /   /                                    //
	//  ____|   | svec[i]                            //
	//   line                                        //
	///////////////////////////////////////////////////
	halfline = (arg.radius + 1) / 2;
	int s2 = Index_ND_Neig_NM(id, dir1, halfline);
    //2 comp, 1st upway 2nd foward 8/9
	for(int il = 0; il < 2; il++){
		int ids=s2;
		msun link0=msun::identity();
		for(int i1=0; i1<l1; i1++){
		link0*= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		ids = Index_ND_Neig_NM(ids, dmu[il], 1);
		}
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dmu1[il], 1);
			}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			ids = Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i1=0; i1<l1; i1++){
			ids = Index_ND_Neig_NM(ids, dmu[il], -1);
 			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		}
		mop += line_left * link0;
	}
	//2 comp, 1st upway 2nd backward 10/11
	for(int il = 0; il < 2; il++){
		int ids=s2;
		msun link0=msun::identity();
		for(int i1=0; i1<l1;i1++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dmu[il], 1);
		}
		for(int i2=0; i2<l2; i2++){
			ids=Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dmu1[il],1);
		}
		for(int i1=0; i1<l1; i1++){
			ids=Index_ND_Neig_NM(ids, dmu[il],-1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		mop += line_left * link0;
	}
	//2 comp, 1st downway 2nd foward 12/13
	for(int il = 0; il < 2; il++){
		int ids=s2;
		msun link0=msun::identity();
		for(int i1=0; i1<l1; i1++){
			ids=Index_ND_Neig_NM(ids, dmu[il],-1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dmu1[il], 1);
		}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			ids= Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i1=0; i1<l1; i1++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids+ dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dmu[il],1);
		}
		mop += line_left * link0;
	}
	//2 comp, 1st downway 2nd backward 6/7
	for(int il = 0; il < 2; il++){
		int ids=s2;
		msun link0=msun::identity();
		for(int i1=0; i1<l1; i1++){
			ids=Index_ND_Neig_NM(ids, dmu[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int i2=0; i2<l2; i2++){
			ids = Index_ND_Neig_NM(ids, dmu1[il], -1);
			link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		}
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + muvolume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids , dir1, 1);
		}
		for(int i2=0; i2<l2; i2++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dmu1[il], 1);
		}
		for(int i1=0; i1<l1; i1++){
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
			ids = Index_ND_Neig_NM(ids, dmu[il], 1);
		}
		mop += line_left * link0;
	}
	}// end of block containing the main part
	mop *= 0.25;
	return mop;
	}

}
