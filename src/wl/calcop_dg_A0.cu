
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

#include <meas/wloopex.h>
#include<vector>
////// operators //////
#include<mop/wloop_arg.h>
#include<mop/opn1.cuh>
#include<mop/opn2.cuh>
#include<mop/opn3.cuh>
#include<mop/mo0.cuh>
#include<mop/mo1.cuh>
#include<mop/mo2.cuh>
#include<mop/mo3.cuh>
#include<mop/mo4.cuh>
//////////////////////
using namespace std;
namespace CULQCD{

template<bool UseTex, class Real, ArrayType atype>
__global__ void kernel_CalcOPsF_A0_33(WLOPArg<Real> arg){
  	int id = INDEX1D();
	if(id >= DEVPARAMS::Volume) return;
	int x[4];
	Index_4D_NM(id, x);
	int muvolume = arg.mu * DEVPARAMS::Volume;
	//int gfoffset = arg.opN * DEVPARAMS::Volume;

	int gfoffset1 = arg.opN * DEVPARAMS::Volume;
	msun link = msun::identity();
	for(int r = 0; r < arg.radius; r++){
		int idx = Index_4D_Neig_NM(x, arg.mu, r);
		link *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, idx + muvolume, DEVPARAMS::size);
	}
	if(arg.opN == 1){
        GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, DEVPARAMS::Volume);
        return;
    }
    // here is just for simple loop
	else GAUGE_SAVE<SOA, Real>( arg.fieldOp, link, id, gfoffset1);
/*
	//COMMON PART
	int halfline = (arg.radius + 1) / 2;
	msun line_left = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, id + muvolume, DEVPARAMS::size);
	for(int ir = 1; ir < halfline; ++ir) 
		line_left *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, ir) + muvolume, DEVPARAMS::size);

	halfline = arg.radius/2;
	msun line_right = GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, halfline) + muvolume, DEVPARAMS::size);
	for(int ir = halfline + 1; ir < arg.radius; ++ir)
		line_right *= GAUGE_LOAD<UseTex, atype, Real>( arg.gaugefield, Index_4D_Neig_NM(id, arg.mu, ir) + muvolume, DEVPARAMS::size);
// extended stape
*/ 
// the above part is not useful for when series N operators have been deactivated.

int index=1;
//template<bool UseTex, class Real, ArrayType atype> DEVICE msun N1( WLOPArg<Real> arg,int id, int lx, int muvolume, msun line_left, msun line_right)
/*
{
	msun mop;
	const int n=4;
	static int len_N1[n]={1,2,3,4};
	#pragma unroll
	for(int i=0; i<n; i++){
		mop=N1< UseTex, Real, atype>(arg, id, len_N1[i], muvolume, line_left, line_right);
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id +index*DEVPARAMS::Volume, gfoffset1);
		index++;
	}
}
*/
//template<bool UseTex, class Real, ArrayType atype> 
//DEVICE msun N2( WLOPArg<Real> arg,int id, int l1, int l2, int muvolume, msun line_left, msun line_right)
/*
{
	msun mop;
	const int n=6;
	static int len1_N2[n]={1,1,2,2,1,3};
	static int len2_N2[n]={1,2,1,2,3,1};
	#pragma unroll
	for(int i=0; i<n; i++){
		mop=N2<UseTex, Real,atype>(arg, id, len1_N2[i], len2_N2[i], muvolume, line_left, line_right);
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index * DEVPARAMS::Volume, gfoffset1);
		index++;
	}
}
*/
//template<bool UseTex, class Real, ArrayType atype> 
//DEVICE msun N3( WLOPArg<Real> arg,int id, int l1, int muvolume, msun line_left, msun line_right)
/*
{
	msun mop;
	const int n=3;
	static int len_N3[n]={1,2,3};
	#pragma unroll
	for(int i=0; i<n; i++){
		mop=N3<UseTex, Real, atype>(arg, id, len_N3[i], muvolume, line_left, line_right);
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index * DEVPARAMS::Volume, gfoffset1);
		index++;
	}
}
*/
// this part is for set MO0
{
msun mop;
const int n=8;
static int set0_l[n]={1, 2, 3, 4, 5, 6,7,8};
#pragma unroll
for(int i=0;i<n;i++){
	mop=MO0<UseTex, Real, atype>(arg, id, set0_l[i],  muvolume);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index * DEVPARAMS::Volume, gfoffset1);
	index++;
}
}
//
///////////////////////////
// this part is for set2 MO1
/*
{
msun mop;
const int n=20;
static int set1_lx[n]={1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,5,6,6,6,7};
static int set1_ly[n]={1,1,2,1,2,3,1,2,3,4,1,2,3,3,4,5,1,2,3,1};
//2,5,5,8,10,10,13,13,18,17,17,20,20,25,25,32,26,26,29,29,34,34,41,41, 37, 37, 39, 39,45,45
#pragma unroll
for(int i=0;i< n;i++){
	mop=MO1<UseTex, Real, atype>(arg, id, set1_lx[i], set1_ly[i],  muvolume);
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index * DEVPARAMS::Volume, gfoffset1);
	index++;
}
}
*/

/*
//////////////////////////////
// this part is for set3 MO2
{
msun mop;
const int n=9;
static int set2_l1x[n]={1,1,2,1,1,2,1,3,2};
static int set2_ly[n]= {1,1,1,2,3,1,1,1,2};
static int set2_l2x[n]={1,2,1,1,1,2,3,1,2};
#pragma unroll
for(int i=0;i<n;i++){
	mop=MO2<UseTex, Real, atype>(arg, id, set2_l1x[i], set2_ly[i], set2_l2x[i],  muvolume); 
	GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index* DEVPARAMS::Volume, gfoffset1);
	index++;
}
}

*/

//template<bool UseTex, class Real, ArrayType atype> 
//DEVICE msun MO4(WLOPArg<Real> arg,int id, int lx, int ly, msun link,int muvolume)
/*
{
	msun mop;
	const int n=3;
	static int set4_l1[n]={1,2,3};
	static int set4_l2[n]={1,2,3};
	#pragma unroll
	for(int i=0; i<n; i++){
		mop=MO4<UseTex, Real, atype>(arg, id, set4_l1[i] , set4_l2[i] ,link , muvolume);
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index* DEVPARAMS::Volume, gfoffset1);
		index++;
	}
}
*/
//DEVICE msun MO3(WLOPArg<Real> arg,int id, int lx1, int ly,int lx2,msun link, int muvolume)
/*
{
	msun mop;
	const int n=3;
	static int set3_l1[n]={1,1,1};
	static int set3_l2[n]={1,1,2};
	static int set3_l3[n]={1,2,1};
	#pragma unroll
	for(int i=0; i<n; i++){
		mop=MO3<UseTex, Real, atype>(arg, id, set3_l1[i], set3_l2[i],set3_l3[i],link, muvolume);
		GAUGE_SAVE<SOA, Real>( arg.fieldOp, mop, id + index* DEVPARAMS::Volume, gfoffset1);
		index++;
	}
}
*/
}
//////////////////////////////////////
// end of kernal for calculation of //
// operators                        //
//////////////////////////////////////
template <bool UseTex, class Real, ArrayType atype> 
class CalcOPsF_A0: Tunable{
private:
   WLOPArg<Real> arg;
	gauge array;
	gauge fieldOp;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer CalcOPsF_A0time;
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
  	kernel_CalcOPsF_A0_33<UseTex, Real, atype><<<tp.grid,tp.block, 0, stream>>>(arg);
}
public:
   CalcOPsF_A0(WLOPArg<Real> arg, gauge array, gauge fieldOp): arg(arg), array(array), fieldOp(fieldOp){
	size = 1;
	for(int i=0;i<4;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;  
}
   ~CalcOPsF_A0(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    CalcOPsF_A0time.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    //CUDA_SAFE_CALL(cudaMemcpy(chromofield, arg.field, 6 * arg.nx * arg.ny * sizeof(Real), cudaMemcpyDeviceToHost));
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    CalcOPsF_A0time.stop();
    timesec = CalcOPsF_A0time.getElapsedTimeInSec();
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
   void stat(){	COUT << "CalcOPsF_A0:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
void CalcWLOPs_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu){
  Timer mtime;
  mtime.start(); 
  WLOPArg<Real> argK;
	argK.gaugefield = array.GetPtr();
	argK.fieldOp = arg->fieldOp.GetPtr();
	argK.radius = radius;
	argK.mu = mu;
	argK.opN = arg->opN;

  
  if(array.Type() != SOA || arg->fieldOp.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true || arg->fieldOp.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
    
  CalcOPsF_A0<UseTex, Real, SOA> wl(argK, array, arg->fieldOp);
  wl.Run();
  if (getVerbosity() >= VERBOSE) wl.stat();
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  if (getVerbosity() >= VERBOSE) COUT << "Time CalcOPsF_A0:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}




template<class Real>
void CalcWLOPs_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu){
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalcWLOPs_A0<true, Real>(array, arg, radius, mu);
  }
  else CalcWLOPs_A0<false, Real>(array, arg, radius, mu);
}


template void CalcWLOPs_A0<double>(gauged array, Sigma_g_plus<double> *arg, int radius, int mu);

}

