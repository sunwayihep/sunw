
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>



#include <timer.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <constants.h>
#include <matrixsun.h>
#include <gaugearray.h>
#include <index.h>
#include <device_PHB_OVR.h>
#include <reunitlink.h>
#include <staple.h>
#include <comm_mpi.h>
#include <exchange.h>
#include <texture_host.h>

#include <sharedmemtypes.h>

#include <tune.h>
#include <launch_kernel.cuh>


#include <cudaAtomic.h>

#include <cub/cub.cuh>



using namespace std;


namespace CULQCD{


inline  __host__   __device__  int periodic(const int id, const int b){
	return (id+b)%b;
}


//From normal to normal lattice index
inline  __host__   __device__ void get4DFL(int id, int x[4]){
  x[3] = id/(param_Grid(0) * param_Grid(1) * param_Grid(2));
  x[2] = (id/(param_Grid(0) * param_Grid(1))) % param_Grid(2);
  x[1] = (id/param_Grid(0)) % param_Grid(1);
  x[0] = id % param_Grid(0); 
}
inline  __host__   __device__ int neighborIndexFL(int id, int mu, int lmu, int nu, int lnu){
	int x[4];
	get4DFL(id, x);
  x[mu] = periodic(x[mu]+lmu, param_Grid(mu));
  x[nu] = periodic(x[nu]+lnu, param_Grid(nu));
  return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0];
}
inline  __host__   __device__ int neighborIndexFL(int id, int mu, int lmu){
	int x[4];
	get4DFL(id, x); 
  x[mu] = periodic(x[mu]+lmu, param_Grid(mu));
  return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1] ) * param_Grid(0) + x[0];
}

__device__ __host__ inline int linkIndex2(int x[], int dx[]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + param_Grid(i)) % param_Grid(i);
  return (((y[3]*param_Grid(2) + y[2])*param_Grid(1) + y[1])*param_Grid(0) + y[0]);
}


__device__ __host__ inline int linkIndex2(int x[], int mu, int lmu) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = x[i];
  y[mu] = periodic(x[mu]+lmu, param_Grid(mu));
  return (((y[3]*param_Grid(2) + y[2])*param_Grid(1) + y[1])*param_Grid(0) + y[0]);
}





template<class Real>
struct WLOPArg{
  complex *gaugefield;
  complex *fieldOp;
  int radius;
  int mu;
  int opN;
};





template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_CalcOPsF_33(WLOPArg<Real> arg){
  	int id = INDEX1D();
	if(id >= DEVPARAMS::Volume) return;
	int x[4];
	get4DFL(id, x);
	int muvolume = arg.mu * DEVPARAMS::Volume;
	//int gfoffset = arg.opN * DEVPARAMS::Volume;

	int gfoffset1 = arg.opN * DEVPARAMS::Volume;
	msun link = msun::identity();
	for(int r = 0; r < arg.radius; r++){
		int idx = linkIndex2(x, arg.mu, r);
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
	}
	if(arg.opN == 1){
        GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, DEVPARAMS::Volume);
        return;
    }
	else GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, gfoffset1);

{
  const int Nbase = 6;
	msun mop[Nbase];
	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//  _____                                        //
	//  |   |____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int ids[2]; int dmu[2];
	ids[0] = linkIndex2(x, dir2, 1);
	ids[1] = linkIndex2(x, dir3, 1);
	dmu[0] = dir2 * DEVPARAMS::Volume;
	dmu[1] = dir3 * DEVPARAMS::Volume;
	int halfline = arg.radius/2;
	//COMMON PART
	msun line = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, arg.radius / 2) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir)
		line *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + dmu[il], DEVPARAMS::size);
		link = link0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
        if(il==0) for(int np = 0; np < Nbase; np++) mop[np]=link;
        else {
            mop[0] += link;
            mop[1] += timesI(link);
            mop[2] -= link;
            mop[3] += link;
            mop[4] += timesI(link);
            mop[5] -= link;
        }
	}
	ids[0] = linkIndex2(x, dir2, -1);
	ids[1] = linkIndex2(x, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		link = link0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
        if(il==0) {
            mop[0] += link;
            mop[1] -= link;
            mop[2] += link;
            mop[3] += link;
            mop[4] -= link;
            mop[5] += link;
        }
        else {
            mop[0] += link;
            mop[1] -= timesI(link);
            mop[2] -= link;
            mop[3] += link;
            mop[4] -= timesI(link);
            mop[5] -= link;
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
	halfline = (arg.radius + 1) / 2;
	int s2 = linkIndex2(x, dir1, halfline);
	ids[0] = neighborIndexFL(s2, dir2, 1);
	ids[1] = neighborIndexFL(s2, dir3, 1);
	//COMMON PART
	line = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, arg.radius) + dmu[il], DEVPARAMS::size);
		link = line * link0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
        if(il==0) {
            mop[0] += link;
            mop[1] += link;
            mop[2] += link;
            mop[3] -= link;
            mop[4] -= link;
            mop[5] -= link;
        }
        else {
            mop[0] += link;
            mop[1] += timesI(link);
            mop[2] -= link;
            mop[3] -= link;
            mop[4] -= timesI(link);
            mop[5] += link;
        }
	}
	ids[0] = neighborIndexFL(s2, dir2, -1);
	ids[1] = neighborIndexFL(s2, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		link = line * link0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
        if(il==0) {
            mop[0] += link;
            mop[1] -= link;
            mop[2] += link;
            mop[3] -= link;
            mop[4] += link;
            mop[5] -= link;
        }
        else {
            mop[0] += link;
            mop[1] -= timesI(link);
            mop[2] -= link;
            mop[3] -= link;
            mop[4] += timesI(link);
            mop[5] += link;
        }
	}
    for(int np = 0; np < Nbase; np++) mop[np]/=(2.0*sqrt(2.0));
    //for(int np = 0; np < Nbase; np++) GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[np], id + (np+1) * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[0], id + DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[1], id + 9 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[2], id + 17 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[3], id + 5 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[4], id + 13 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[5], id + 21 * DEVPARAMS::Volume, gfoffset1);



}





{
  const int Nbase = 12;
	msun mop[Nbase];
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
	int ids[2]; int dmu[2]; int dmu1[2];
	ids[0] = linkIndex2(x, dir2, 1);
	ids[1] = linkIndex2(x, dir3, 1);
	dmu[0] = dir2;
	dmu[1] = dir3;
	dmu1[0] = dir3;
	dmu1[1] = dir2;
	int halfline = arg.radius/2;
	//COMMON PART
	msun line = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir)
		line *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, 1st upway 2nd foward  0/1
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(id, dmu[il], 1, dmu1[il], 1);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
    ids[il] = neighborIndexFL(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = link0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);

    if(il==0){ for(int np = 0; np < Nbase; np++) mop[np]=link;}
    else{
            mop[0] += link;
            mop[1] -= link;
            mop[2] += link;
            mop[3] -= link;
            mop[4] += timesI(link);
            mop[5] -= timesI(link);
            mop[6] += timesI(link);
            mop[7] -= timesI(link);
            mop[8] -= link;
            mop[9] += link;
            mop[10] -= link;
            mop[11] += link;
    }
	}
	//2 comp, 1st upway 2nd backward  2/3
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(id, dmu[il], 1, dmu1[il], -1);
    link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);

		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = link0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] -= link;
            mop[2] += link;
            mop[3] -= link;
            mop[4] += link;
            mop[5] -= link;
            mop[6] += link;
            mop[7] -= link;
            mop[8] += link;
            mop[9] -= link;
            mop[10] += link;
            mop[11] -= link;
    }
    else{
            mop[0] += link;
            mop[1] += link;
            mop[2] += link;
            mop[3] += link;
            mop[4] += timesI(link);
            mop[5] += timesI(link);
            mop[6] += timesI(link);
            mop[7] += timesI(link);
            mop[8] -= link;
            mop[9] -= link;
            mop[10] -= link;
            mop[11] -= link;
    }
	}

	ids[0] = linkIndex2(x, dir2, -1);
	ids[1] = linkIndex2(x, dir3, -1);
	//2 comp, 1st downway 2nd foward 4/5
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(ids[il], dmu1[il], 1);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
    ids[il] = neighborIndexFL(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = link0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] -= link;
            mop[2] += link;
            mop[3] -= link;
            mop[4] -= link;
            mop[5] += link;
            mop[6] -= link;
            mop[7] += link;
            mop[8] += link;
            mop[9] -= link;
            mop[10] += link;
            mop[11] -= link;
    }
    else{
            mop[0] += link;
            mop[1] += link;
            mop[2] += link;
            mop[3] += link;
            mop[4] -= timesI(link);
            mop[5] -= timesI(link);
            mop[6] -= timesI(link);
            mop[7] -= timesI(link);
            mop[8] -= link;
            mop[9] -= link;
            mop[10] -= link;
            mop[11] -= link;
    }
	}

	ids[0] = linkIndex2(x, dir2, -1);
	ids[1] = linkIndex2(x, dir3, -1);
	//2 comp, 1st downway 2nd backward 6/7
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(ids[il], dmu1[il], -1);
    link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(ids[il], dmu1[il], 1);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = link0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] += link;
            mop[2] += link;
            mop[3] += link;
            mop[4] -= link;
            mop[5] -= link;
            mop[6] -= link;
            mop[7] -= link;
            mop[8] += link;
            mop[9] += link;
            mop[10] += link;
            mop[11] += link;
    }
    else{
            mop[0] += link;
            mop[1] -= link;
            mop[2] += link;
            mop[3] -= link;
            mop[4] -= timesI(link);
            mop[5] += timesI(link);
            mop[6] -= timesI(link);
            mop[7] += timesI(link);
            mop[8] -= link;
            mop[9] += link;
            mop[10] -= link;
            mop[11] += link;
    }
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
	int s2 = linkIndex2(x, dir1, halfline);
	ids[0] = neighborIndexFL(s2, dir2, 1);
	ids[1] = neighborIndexFL(s2, dir3, 1);
	//COMMON PART
	line = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
    //2 comp, 1st upway 2nd foward 8/9
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(s2, dmu[il], 1);
    link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(ids[il], dmu1[il], 1);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
    ids[il] = neighborIndexFL(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(ids[il], dmu[il], -1);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = line * link0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] += link;
            mop[2] -= link;
            mop[3] -= link;
            mop[4] += link;
            mop[5] += link;
            mop[6] -= link;
            mop[7] -= link;
            mop[8] += link;
            mop[9] += link;
            mop[10] -= link;
            mop[11] -= link;
    }
    else{
            mop[0] += link;
            mop[1] -= link;
            mop[2] -= link;
            mop[3] += link;
            mop[4] += timesI(link);
            mop[5] -= timesI(link);
            mop[6] -= timesI(link);
            mop[7] += timesI(link);
            mop[8] += link;
            mop[9] += link;
            mop[10] += link;
            mop[11] -= link;
    }
	}
	//2 comp, 1st upway 2nd backward 10/11
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(s2, dmu[il], 1, dmu1[il], -1);
    link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
    ids[il] = neighborIndexFL(ids[il], dmu[il], -1, dmu1[il], 1);
 		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
 		//link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, arg.radius) + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = line * link0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] -= link;
            mop[2] -= link;
            mop[3] += link;
            mop[4] += link;
            mop[5] -= link;
            mop[6] -= link;
            mop[7] += link;
            mop[8] -= link;
            mop[9] -= link;
            mop[10] -= link;
            mop[11] += link;
    }
    else{
            mop[0] += link;
            mop[1] += link;
            mop[2] -= link;
            mop[3] -= link;
            mop[4] += timesI(link);
            mop[5] += timesI(link);
            mop[6] -= timesI(link);
            mop[7] -= timesI(link);
            mop[8] -= link;
            mop[9] -= link;
            mop[10] += link;
            mop[11] += link;
    }
	}
	ids[0] = neighborIndexFL(s2, dir2, -1);
	ids[1] = neighborIndexFL(s2, dir3, -1);
	//2 comp, 1st downway 2nd foward 12/13
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        ids[il] = neighborIndexFL(ids[il], dmu1[il], 1);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
        ids[il] = neighborIndexFL(ids[il], dmu1[il], -1);
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = line * link0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] -= link;
            mop[2] -= link;
            mop[3] += link;
            mop[4] -= link;
            mop[5] += link;
            mop[6] += link;
            mop[7] -= link;
            mop[8] += link;
            mop[9] -= link;
            mop[10] -= link;
            mop[11] += link;
    }
    else{
            mop[0] += link;
            mop[1] += link;
            mop[2] -= link;
            mop[3] -= link;
            mop[4] -= timesI(link);
            mop[5] -= timesI(link);
            mop[6] += timesI(link);
            mop[7] += timesI(link);
            mop[8] -= link;
            mop[9] -= link;
            mop[10] += link;
            mop[11] += link;
    }
	}
	ids[0] = neighborIndexFL(s2, dir2, -1);
	ids[1] = neighborIndexFL(s2, dir3, -1);
	//2 comp, 1st downway 2nd backward 14/15
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        ids[il] = neighborIndexFL(ids[il], dmu1[il], -1);
        link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu1[il] * DEVPARAMS::Volume, DEVPARAMS::size);
        ids[il] = neighborIndexFL(ids[il], dmu1[il], 1);
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il] * DEVPARAMS::Volume, DEVPARAMS::size);
		msun link = line * link0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0){
            mop[0] += link;
            mop[1] += link;
            mop[2] -= link;
            mop[3] -= link;
            mop[4] -= link;
            mop[5] -= link;
            mop[6] += link;
            mop[7] += link;
            mop[8] += link;
            mop[9] += link;
            mop[10] -= link;
            mop[11] -= link;
    }
    else{
            mop[0] += link;
            mop[1] -= link;
            mop[2] -= link;
            mop[3] += link;
            mop[4] -= timesI(link);
            mop[5] += timesI(link);
            mop[6] += timesI(link);
            mop[7] -= timesI(link);
            mop[8] -= link;
            mop[9] += link;
            mop[10] += link;
            mop[11] -= link;
    }
	}
    for(int np = 0; np < Nbase; np++) mop[np]*= 0.25;
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[0], id + 2 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[1], id + 4 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[2], id + 6 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[3], id + 8 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[4], id + 10 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[5], id + 11 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[6], id + 14 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[7], id + 15 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[8], id + 18 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[9], id + 19 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[10], id + 22 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[11], id + 23 * DEVPARAMS::Volume, gfoffset1);
}






{
  const int Nbase = 6;
	msun mop[Nbase];


	///////////////////////////////////////////////////
	//                                               //
	//  CALCULATE PATHS OF THE TYPE:                 //
	//                                               //
	//  _____                                        //
	//  |___|____ line                               //
	//  svec[i]                                      //
	///////////////////////////////////////////////////
	int dir1 = arg.mu;
	int dir2 = (dir1+1)%3;
	int dir3 = (dir1+2)%3;
	int ids[2]; int dmu[2];
	ids[0] = linkIndex2(x, dir2, 1);
	ids[1] = linkIndex2(x, dir3, 1);
	dmu[0] = dir2 * DEVPARAMS::Volume;
	dmu[1] = dir3 * DEVPARAMS::Volume;
	int halfline = arg.radius/2;
	//COMMON PART
	msun line = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir)
		line *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
    msun line0= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + dmu[il], DEVPARAMS::size);
		msun link = line0 * link0.dagger() * line0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0) for(int np = 0; np < Nbase; np++) mop[np]=link;
    else {
        mop[0] += link;
        mop[1] += link;
        mop[2] += timesI(link);;
        mop[3] += timesI(link);;
        mop[4] -= link;
        mop[5] -= link;
    }



	}
	ids[0] = linkIndex2(x, dir2, -1);
	ids[1] = linkIndex2(x, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = 0; ir < halfline; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		msun link = line0 * link0.dagger() * line0 * line;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0) {
        mop[0] += link;
        mop[1] += link;
        mop[2] -= link;
        mop[3] -= link;
        mop[4] += link;
        mop[5] += link;
    }
    else {
        mop[0] += link;
        mop[1] += link;
        mop[2] -= timesI(link);;
        mop[3] -= timesI(link);;
        mop[4] -= link;
        mop[5] -= link;
    }
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
	int s2 = linkIndex2(x, dir1, halfline);
	ids[0] = neighborIndexFL(s2, dir2, 1);
	ids[1] = neighborIndexFL(s2, dir3, 1);
	//COMMON PART
	line = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
    line0= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir) 
		line0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, ir) + muvolume, DEVPARAMS::size);
	//2 comp, in upway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, s2 + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, neighborIndexFL(id, dir1, arg.radius) + dmu[il], DEVPARAMS::size);
		msun link = line * line0 * link0.dagger() * line0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0) {
        mop[0] += link;
        mop[1] -= link;
        mop[2] += link;
        mop[3] -= link;
        mop[4] += link;
        mop[5] -= link;
    }
    else {
        mop[0] += link;
        mop[1] -= link;
        mop[2] += timesI(link);;
        mop[3] -= timesI(link);;
        mop[4] -= link;
        mop[5] += link;
    }
	}
	ids[0] = neighborIndexFL(s2, dir2, -1);
	ids[1] = neighborIndexFL(s2, dir3, -1);
	//2 comp, in downway
	for(int il = 0; il < 2; il++){		
		msun link0 = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		for(int ir = halfline; ir < arg.radius; ++ir) {
			link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + muvolume, DEVPARAMS::size);
			ids[il] = neighborIndexFL(ids[il], dir1, 1);
		}
		link0 *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, ids[il] + dmu[il], DEVPARAMS::size);
		msun link = line * line0 * link0.dagger() * line0;
		//GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id + (iop + il) * DEVPARAMS::Volume, gfoffset);
    if(il==0) {
        mop[0] += link;
        mop[1] -= link;
        mop[2] -= link;
        mop[3] += link;
        mop[4] += link;
        mop[5] -= link;
    }
    else {
        mop[0] += link;
        mop[1] -= link;
        mop[2] -= timesI(link);
        mop[3] += timesI(link);
        mop[4] -= link;
        mop[5] += link;
    }
	}
    for(int np = 0; np < Nbase; np++) mop[np]/=(2.0*sqrt(2.0));
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[0], id + 3 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[1], id + 7 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[2], id + 12 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[3], id + 16 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[4], id + 20 * DEVPARAMS::Volume, gfoffset1);
    GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop[5], id + 24 * DEVPARAMS::Volume, gfoffset1);
}

}


template <bool UseTex, class Real, ArrayType atype> 
class CalcOPsF: Tunable{
private:
   WLOPArg<Real> arg;
	gauge array;
	gauge fieldOp;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer CalcOPsFtime;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  //CUDA_SAFE_CALL(cudaMemset(arg.fieldOp, 0, PARAMS::Volume * arg.opN * sizeof(msun)));
	fieldOp.Clean();
  	kernel_CalcOPsF_33<UseTex, Real, atype><<<tp.grid,tp.block, 0, stream>>>(arg);
}
public:
   CalcOPsF(WLOPArg<Real> arg, gauge array, gauge fieldOp): arg(arg), array(array), fieldOp(fieldOp){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
   ~CalcOPsF(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    CalcOPsFtime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    //CUDA_SAFE_CALL(cudaMemcpy(chromofield, arg.field, 6 * arg.nx * arg.ny * sizeof(Real), cudaMemcpyDeviceToHost));
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    CalcOPsFtime.stop();
    timesec = CalcOPsFtime.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const {
	long long tmp = (arg.radius - arg.radius/2 - 2LL + 4LL * (arg.radius / 2 + 1LL)) * array.getNumFlop(true) + 4LL * array.getNumFlop(false) + (arg.radius - arg.radius/2 - 1LL + 4LL * arg.radius / 2 + 4LL) * 198LL;
	tmp *= 2LL;
	tmp += arg.radius * (array.getNumFlop(true) + 198LL) + array.getNumFlop(false);
	tmp *= PARAMS::Volume;
	return tmp;}
   long long bytes() const{ 
		long long tmp = (arg.radius - arg.radius/2 - 2LL + 4LL * (arg.radius / 2 + 2LL)) * array.getNumParams() * sizeof(Real);
		tmp *= 2LL;
		tmp += (arg.radius + 1LL) * array.getNumParams() * sizeof(Real);
		tmp *= PARAMS::Volume;
return tmp;}
   double time(){	return timesec;}
   void stat(){	COUT << "CalcOPsF:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() {  }

};






template<bool UseTex, class Real>
void CalcWLOPs_dg_33(gauge array, gauge fieldOp, int radius, int mu, int opN){
  Timer mtime;
  mtime.start(); 
  WLOPArg<Real> arg;
	arg.gaugefield = array.GetPtr();
	arg.fieldOp = fieldOp.GetPtr();
	arg.radius = radius;
	arg.mu = mu;
	arg.opN = opN;

  
  if(array.Type() != SOA || fieldOp.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || fieldOp.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
    
  CalcOPsF<UseTex, Real, SOA> wl(arg, array, fieldOp);
  wl.Run();
  if (getVerbosity() >= VERBOSE) wl.stat();
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  if (getVerbosity() >= VERBOSE) COUT << "Time CalcOPsF:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




template<class Real>
void CalcWLOPs_dg_33(gauge array, gauge fieldOp, int radius, int mu, int opN){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalcWLOPs_dg_33<true, Real>(array, fieldOp, radius, mu, opN);
  }
  else CalcWLOPs_dg_33<false, Real>(array, fieldOp, radius, mu, opN);
}


//template void CalcWLOPs_dg_33<float>(gauges array, gauges fieldOp, int radius, int mu, int opN);
template void CalcWLOPs_dg_33<double>(gauged array, gauged fieldOp, int radius, int mu, int opN);










}
