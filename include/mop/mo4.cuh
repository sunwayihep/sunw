#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
	
//////////////////////////////////////
//          /|         /|           //
//         / |        / | ly        //
//         | /        | /           //
//         |/_________|/ lx         //
//////////////////////////////////////
//////////////////////////////////////////////////////
//  ______________                                  //
// |      |       |                                 //
// |  2   |   1   | the loops in each area is has a //
// |______|_______| index of that area, for example //
// |      |       | MO41 sites in 1st area. quark is//
// |  3   |   4   | in the center. all the loops has//
// |______|_______| been written in anti_clock_wise.//
//////////////////////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO4_q1(WLOPArg<Real> arg,int id,int dir1,int dir2, int lx,int ly){
	int dmu[2]={dir1,dir2};
	int ids=id;
	msun link;
	link=msun::identity();
	for(int ix=0;ix<lx;ix++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[0],1);
	}
	for(int iy=0; iy<ly;iy++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[1],1);
	}
	for(int ix=0;ix<lx;ix++){
		ids=Index_ND_Neig_NM(ids,dmu[0],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int iy=0; iy<ly;iy++){
		ids=Index_ND_Neig_NM(ids,dmu[1],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	return link;
}
///////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO4_q2(WLOPArg<Real> arg,int id,int dir1, int dir2, int lx,int ly){
	int dmu[2]={dir1, dir2};
	int ids=id;
	msun link;
	link=msun::identity();
	// to keep it anticlockwise
	for(int iy=0; iy<ly;iy++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[1],1);
	}
	for(int ix=0;ix<lx;ix++){
		ids=Index_ND_Neig_NM(ids,dmu[0],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int iy=0; iy<ly;iy++){
		ids=Index_ND_Neig_NM(ids,dmu[1],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int ix=0;ix<lx;ix++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[0],1);
	}
	return link;
}
//////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO4_q3(WLOPArg<Real> arg,int id,int dir1, int dir2, int lx,int ly){
	int dmu[2]={dir1, dir2};
	int ids=id;
	msun link;
	link=msun::identity();
	// to keep it anticlockwise
	for(int ix=0;ix<lx;ix++){
		ids=Index_ND_Neig_NM(ids,dmu[0],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int iy=0; iy<ly;iy++){
		ids=Index_ND_Neig_NM(ids,dmu[1],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int ix=0;ix<lx;ix++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[0],1);
	}
	for(int iy=0; iy<ly;iy++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[1],1);
	}
	return link;
}
/////////////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO4_q4(WLOPArg<Real> arg,int id,int dir1, int dir2, int lx,int ly){
	int dmu[2]={dir1, dir2};
	int ids=id;
	msun link;
	link=msun::identity();
	// to keep it anticlockwise
	for(int iy=0; iy<ly;iy++){
		ids=Index_ND_Neig_NM(ids,dmu[1],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int ix=0;ix<lx;ix++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[0],1);
	}
	for(int iy=0; iy<ly;iy++){
		link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids=Index_ND_Neig_NM(ids,dmu[1],1);
	}
	for(int ix=0;ix<lx;ix++){
		ids=Index_ND_Neig_NM(ids,dmu[0],-1);
		link*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	return link;
}
// MO4
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO4(WLOPArg<Real> arg,int id, int lx, int ly, msun link ,int muvolume){
	msun mop=msun::zero();
		{
		int dir1=arg.mu;
		int mu[2]={(dir1+1)%3, (dir1+2)%3};
		int ids=Index_ND_Neig_NM(id, dir1, arg.radius);
//		msun link=msun::identity();
//		for(int ir=0;ir<arg.radius;ir++){
//			link*=GAUGE_LOAD< UseTex, atype, Real>(arg.gaugefield, ids+muvolume, DEVPARAMS::size);
//			ids=Index_4D_Neig_NM(ids, dir1, 1);
//		}
		msun qu[4];
		msun aqu[4];
		//MO4_q1(WLOPArg<Real> arg,int id,int dir1,int dir2, int lx,int ly)
		qu[0]=MO4_q1<UseTex,Real, atype>(arg, id, mu[0], mu[1], lx, ly);
		qu[1]=MO4_q2<UseTex,Real, atype>(arg, id, mu[0], mu[1], lx, ly);
		qu[2]=MO4_q3<UseTex,Real, atype>(arg, id, mu[0], mu[1], lx, ly);
		qu[3]=MO4_q4<UseTex,Real, atype>(arg, id, mu[0], mu[1], lx, ly);
		//ids is the antiquark location
		aqu[0]=MO4_q1<UseTex,Real, atype>(arg, ids, mu[0], mu[1], lx, ly);
		aqu[1]=MO4_q2<UseTex,Real, atype>(arg, ids, mu[0], mu[1], lx, ly);
		aqu[2]=MO4_q3<UseTex,Real, atype>(arg, ids, mu[0], mu[1], lx, ly);
		aqu[3]=MO4_q4<UseTex,Real, atype>(arg, ids, mu[0], mu[1], lx, ly);
		#pragma unroll
		for(int i=0;i<4;i++){
			// loop in the same area and with same orientation
			mop+=qu[i]*link*aqu[i];
			mop+=qu[i].dagger()*link*aqu[i].dagger();
			// loop in the same area and with oposit orientation
			mop+=qu[i]*link*aqu[i].dagger();
			mop+=qu[i].dagger()*link*aqu[i];
			// loop in the opposite area and with the same orientation
			mop+=qu[i]*link*aqu[(i+2)%4];
			mop+=qu[i].dagger()*link*aqu[(i+2)%4].dagger();
			// loop in the opposit area and with opposit orientation
			mop+=qu[i]*link*aqu[(i+2)%4].dagger();
			mop+=qu[i].dagger()*link*aqu[(i+2)%4];
		}
	}
	mop/=(4.0*sqrt(2.0));
	return mop;
}



}
