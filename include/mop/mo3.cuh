#pragma once
#include<mop/wloop_arg.h>
namespace CULQCD{
/////////////////////////////
//   ___________           //
//   |   2*ly   |          //
//	 |____  ____| lx2      // this is front view with assumption that
//		  ||               // quark antiquark is in z-direction.
//		  || lx1           //
/////////////////////////////
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun qac(WLOPArg<Real> arg,int id,int dir1, int dir2, int lx1,int ly, int lx2){
	msun staple;
	{
	int dmu[2];
	dmu[0]=dir1;
	dmu[1]=dir2;
	int ids[2];
	ids[0]=id;
	ids[1]=id;
	msun link[2];
	link[0]=msun::identity();
	link[1]=msun::identity();
	// up side, anti_clock_wise
	//	||
	//	||
	for(int ix1=0;ix1<lx1; ix1++){
		link[0]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield,ids[0]+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size );
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[0], 1);
	}
	link[1]=link[0];
	ids[1]=ids[0];
	// <-- -->
	for(int iy=0;iy<ly;iy++){
		link[0]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield,ids[0]+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size );
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[1], 1);
		ids[1]=Index_4D_Neig_NM(ids[1], dmu[1],-1);
		link[1]*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield,ids[1]+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size ); // this goes backward
	}
	// ^
	// |
	for(int ix2=0; ix2<lx2;ix2++){
		link[0]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield,ids[0]+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size );
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[0], 1);
		link[1]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield,ids[1]+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size );
		ids[1]=Index_4D_Neig_NM(ids[1], dmu[0], 1);
	}
	// -->
	for(int iy=0;iy<2*ly;iy++){
		link[1]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield,ids[1]+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size );
		ids[1]=Index_4D_Neig_NM(ids[1], dmu[1], 1);
	}
	staple=link[0]*link[1].dagger();
	}
	return staple;
}
//////////////////////////////
//			||lx1			//
//		____||____			//
//	   |          |lx2		//
//	   |__________|			//
//	       2*ly				//
//////////////////////////////
// dir1 and dir2 depends in z direction, for example if z =3, dir1=1, dir2=2
// for Mo_4{1,2}, and for Mo_4{1,4}, we have dir1=2, dir1=1
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun qmac(WLOPArg<Real> arg,int id,int dir1, int dir2, int lx1,int ly, int lx2){
	msun staple;
	{
	int dmu[2];
	dmu[0]=dir1;
	dmu[1]=dir2;
	int ids[2];
	ids[0]=id;
	ids[1]=id;
	msun link[2];
	link[0]=msun::identity();
	link[1]=msun::identity();
	// first down, anticlock
	for(int ix1=0; ix1<lx1;ix1++){
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[0], -1);
		link[0]*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
		}
	link[1]=link[0];
	ids[1]=ids[0];
	for(int iy=0; iy<ly;iy++){
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[1],-1);
		link[0]*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		link[1]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids[1]=Index_4D_Neig_NM(ids[1], dmu[1], 1);
	}
	for(int ix2=0;ix2<lx2; ix2++){
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[0],-1);
		link[0]*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids[1]=Index_4D_Neig_NM(ids[1], dmu[0],-1);
		link[1]*=GAUGE_LOAD_DAGGER<UseTex, atype, Real>(arg.gaugefield, ids[1]+dmu[0]*DEVPARAMS::Volume, DEVPARAMS::size);
	}
	for(int iy=0; iy<2*ly;iy++){
		link[0]*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids[0]+dmu[1]*DEVPARAMS::Volume, DEVPARAMS::size);
		ids[0]=Index_4D_Neig_NM(ids[0], dmu[1], 1);
	}
	staple=link[0]*link[1].dagger(); // anti_clock_wise
	}
	return staple;
}
// the following function is make MO3 using qac and qmac
template<bool UseTex, class Real, ArrayType atype> 
DEVICE msun MO3(WLOPArg<Real> arg,int id, int lx1, int ly,int lx2,msun link, int muvolume){
	msun mop=msun::zero();
	{
	int dir1=arg.mu;
	int mu[2]={(dir1+1)%3,(dir1+2)%3};
	int ids=Index_4D_Neig_NM(id, dir1, arg.radius);
//	msun link=msun::identity();
//	for(int ir=0; ir< arg.radius; ir++){
//	link*=GAUGE_LOAD<UseTex, atype, Real>(arg.gaugefield, ids+muvolume, DEVPARAMS::size);
//	ids=Index_4D_Neig_NM(ids, dir1 ,1);
//	}
	//qac(WLOPArg<Real> arg,int id,int dir1, int dir2, int lx1,int ly, int lx2)
	msun qu[4];
	// id is at the quark
	qu[0]=qac<UseTex,Real, atype>(arg , id , mu[0], mu[1] ,lx1 , ly, lx2);//q12
	qu[1]=qac<UseTex,Real, atype>(arg ,id, mu[1] , mu[0], lx1, ly, lx2);//q23
	qu[2]=qmac<UseTex,Real, atype>(arg,id,mu[0],mu[1], lx1,ly,lx2);//q34
	qu[3]=qmac<UseTex,Real, atype>(arg,id,mu[1],mu[0],lx1,ly, lx2);//q14
	msun aqu[4];
	// ids is already end of the line, antiquark
	aqu[0]=qac<UseTex,Real, atype>(arg,ids,mu[0], mu[1],lx1,ly, lx2);
	aqu[1]=qac<UseTex,Real, atype>(arg,ids,mu[1], mu[0],lx1,ly, lx2);
	aqu[2]=qmac<UseTex,Real, atype>(arg,ids,mu[0],mu[1],lx1,ly, lx2);
	aqu[3]=qmac<UseTex,Real, atype>(arg,ids,mu[1],mu[0],lx1,ly, lx2);
	#pragma unroll
	for(int i=0;i<4;i++){
		// this part is for loop in same area and same direction
		mop+=qu[i]*link*aqu[i];
		mop+=qu[i].dagger()*link*aqu[i].dagger();// this is of P_x operator
		////////////////////
		// loop in same area and opposit rotation
		mop+=qu[i]*link*aqu[i].dagger();
		mop+=qu[i].dagger()*link*aqu[i];
		/////////////////////
		// loop in opposit area and same rotation
		mop+=qu[i]*link*aqu[(i+2)%4];
		mop+=qu[i].dagger()*link*aqu[(i+2)%4].dagger();
		/////////////////////
		// loop in opposit area and opposit ratation
		mop+=qu[i]*link*aqu[(i+2)%4].dagger();
		mop+=qu[i].dagger()*link*aqu[(i+2)%4];
		}
	}
	mop/=(4.0*sqrt(2.0));
	return mop;
}

}
