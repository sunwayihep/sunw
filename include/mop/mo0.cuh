#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
//////////////////////////////////////////////////S
//                 Alireza                      //
//      ______                                  //
//     |      |lx                               //
//     |      |   MO=More operator              //
//                                              //
//////////////////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO0(WLOPArg<Real> arg,int id, int lx, int muvolume){
	msun mop = msun::zero();
	int dir1 = arg.mu;
	int dmu[2];
	dmu[0]=(dir1+1)%3;
	dmu[1]=(dir1+2)%3;
	//0 comp 1st upway
	for(int il=0;il<2;il++){
		int ids=id;
		msun link0=msun::identity();
		for(int ix=0; ix<lx;ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dmu[il], 1);
		}
		// dir1 comp
		for(int ir=0;ir<arg.radius;ir++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+muvolume, DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dir1, 1);
		}
		// downway comp
		for(int ix=0;ix<lx;ix++){
			ids=Index_ND_Neig_NM(ids, dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		mop+=link0;
	}
	//1 comp 1st downway
	for(int il=0; il<2; il++){
		int ids=id;
		msun link0=msun::identity();
		// downway comp
		for(int ix=0; ix<lx;ix++){
			ids=Index_ND_Neig_NM(ids, dmu[il], -1);
			link0*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
		//dir1 comp
		for(int ir=0;ir<arg.radius;++ir){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+muvolume, DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dir1, 1);
		}
		//upway comp
		for(int ix=0; ix<lx; ix++){
			link0*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[il]*DEVPARAMS::Volume , DEVPARAMS::size);
			ids=Index_ND_Neig_NM(ids, dmu[il], 1);
		}
		mop+=link0;
	}
	mop/=(2.0);
	return mop;
}

}
