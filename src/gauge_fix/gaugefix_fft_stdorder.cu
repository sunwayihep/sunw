
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <typeinfo>

#include <cufft.h>

#include <comm_mpi.h>
#include <gaugefix/gaugefix.h>
#include <cuda_common.h>
#include <complex.h>
#include <matrixsun.h>
#include <constants.h>
#include <CUFFT_Plans.h>
#include <texture_host.h>
#include <timer.h>

#include <complex.h>
#include <matrixsun.h>
#include <gaugearray.h>
#include <constants.h>
#include <index.h>
#include <reunitlink.h>
#include <device_load_save.h>
#include <reduction.h>

#include <tune.h>
#include <sharedmemtypes.h>

namespace CULQCD{
namespace NormalIdOrder{

#ifndef FL_UNITARIZE_PI
#define FL_UNITARIZE_PI 3.14159265358979323846
#endif


static inline __host__ __device__ void getCoords4(int x[4], const int id, const int X[4]){
	x[3] = id/(X[0] * X[1] * X[2]);
	x[2] = (id/(X[0] * X[1])) % X[2];
	x[1] = (id/X[0]) % X[1];
	x[0] = id % X[0];
}
static inline __host__ __device__ int neighborIndexMinusOne(int x[4], const int mu, const int X[4]){
	int y[4];
	for(int dir=0; dir<4;dir++) y[dir] = x[dir];
	y[mu] = (y[mu] - 1 + X[mu]) % X[mu];
	return (y[0] + (y[1] + (y[2] + y[3] * X[2]) * X[1]) * X[0]);
}


static inline __host__ __device__ int neighborIndexPlusOne(int x[4], const int mu, const int X[4]){
	int y[4];
	for(int dir=0; dir<4;dir++) y[dir] = x[dir];
	y[mu] = (y[mu] + 1) % X[mu];
	return (y[0] + (y[1] + (y[2] + y[3] * X[2]) * X[1]) * X[0]);
}




template <bool UseTex, ArrayType atype, ArrayType atypedelta, class Real, int DIR>
__global__ void  kernel_calc_Fg_theta_delta(complex *array, complex *res_save, complex *Delta){
	int idx = INDEX1D();
	if(idx < DEVPARAMS::Volume){
		int offset = DEVPARAMS::Volume * 4;
		msun delta = msun::zero();
		//Uplinks
		for(int nu = 0; nu < DIR; nu++) 
			delta -= GAUGE_LOAD<UseTex, atype, Real>( array,  idx + nu * DEVPARAMS::Volume, offset);
		complex res;
		//Fg (sum_DIR uplinks)
		res.real() = -delta.realtrace();
		//Downlinks
		int x[4];
		getCoords4(x, idx, DEVPARAMS::Grid);
		for(int nu = 0; nu < DIR; nu++) 	
			delta += GAUGE_LOAD<UseTex, atype, Real>( array, neighborIndexMinusOne(x, nu, DEVPARAMS::Grid) + nu * DEVPARAMS::Volume, offset);
		delta = (delta-delta.dagger()).subtraceunit();
        //Save Delta
        DELTA_SAVE<atypedelta, Real>( Delta, delta , idx, DEVPARAMS::Volume);
		//theta
		res.imag() = realtraceUVdagger(delta, delta);
		res_save[idx] = res;
	}
	//FLOP per lattice site = 2 * NCOLORS * NCOLORS * (DIR + 1) + 4 * NCOLORS * ( 1 + NCOLORS)
	//The FLOP number does not include the reconstruction when used
}







template <bool UseTex, ArrayType atype, ArrayType atypedelta, class Real, int DIR> 
class GaugeFixFFTQuality: Tunable{
private:
   string functionName;
   typedef void (*TFuncPtr)(complex*, complex*, complex*);
   TFuncPtr kernel_pointer;
   gauge array;
   complex *sum;
   gauge delta;
   int size;
   complex value;
   double timesec;
   int grid[4];
#ifdef TIMMINGS
    Timer mtime;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_pointer<<<tp.grid,tp.block, 0, stream>>>(array.GetPtr(), sum, delta.GetPtr());
	}

public:
   GaugeFixFFTQuality(gauge &array, gauge &delta, int gridin[4]):array(array), delta(delta){
		functionName = "GaugeFixFFTQuality";
		value = complex::zero();
		kernel_pointer = NULL;
		if(!array.EvenOdd()){	
			kernel_pointer = &kernel_calc_Fg_theta_delta<UseTex, atype, atypedelta, Real, DIR>;
		}
		if(kernel_pointer == NULL) errorCULQCD("No kernel GaugeFixFFTQuality function exist for this gauge array...");
		size = 1;
		for(int i=0;i<4;i++){
			grid[i]=gridin[i];
			size *= gridin[i];
		} 
		timesec = 0.0;
		sum = (complex*) dev_malloc(size*sizeof(complex));
	}
   GaugeFixFFTQuality(gauge &array, gauge &delta, complex *sum, int gridin[4]):array(array), delta(delta), sum(sum){
		functionName = "GaugeFixFFTQuality";
		value = complex::zero();
		kernel_pointer = NULL;
		if(!array.EvenOdd()){	
			kernel_pointer = &kernel_calc_Fg_theta_delta<UseTex, atype, atypedelta, Real, DIR>;
		}
		if(kernel_pointer == NULL) errorCULQCD("No kernel GaugeFixFFTQuality function exist for this gauge array...");
		size = 1;
		for(int i=0;i<4;i++){
			grid[i]=gridin[i];
			size *= gridin[i];
		} 
		timesec = 0.0;   	
   }
   ~GaugeFixFFTQuality(){ dev_free(sum);};


	complex Run(const cudaStream_t &stream){
	#ifdef TIMMINGS
	    mtime.start();
	#endif
	    apply(stream);
		value = reduction<complex>(sum, size, stream);
		value /= (Real)(PARAMS::Volume * NCOLORS);
		value.real() /= (Real)DIR;
		#ifdef MULTI_GPU
		comm_Allreduce(&value);
		value /= numnodes();
		#endif
	#ifdef TIMMINGS
		CUDA_SAFE_DEVICE_SYNC( );
	    mtime.stop();
	    timesec = mtime.getElapsedTimeInSec();
	#endif
		return value;
	}

	complex Run(){return Run(0);}

	double time(){return timesec;}

	void stat(){
		COUT << "GaugeFixFFTQuality:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;
	}

	void printValue(){
		printfCULQCD("GaugeFixFFTQuality: Fg = %.12e\ttheta = %.12e\n", value.real(), value.imag() );
	}

	long long flop() const { 
		//not acccounting reduction!!!!!
		long long arrayflops = 2LL * DIR * array.getNumFlop(true) + delta.getNumFlop(false);
		return (arrayflops + 2LL * NCOLORS * NCOLORS * (DIR + 1) + 4LL * NCOLORS * ( 1 + NCOLORS) ) * size;
	}

	long long bytes() const { 
		//not acccounting reduction!!!!!
		return (2LL * DIR * array.getNumParams() + delta.getNumParams() + 2LL) * size * sizeof(Real);
	}

	double flops(){	return ((double)flop() * 1.0e-9) / timesec;}

	double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << grid[0] << "x";
    vol << grid[1] << "x";
    vol << grid[2] << "x";
    vol << grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), array.ToStringArrayType().c_str(), aux.str().c_str());
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





template <class Real>
struct GaugeFixArg {
	int threads; // number of active threads required
	int X[4]; // grid dimensions
  	complex *pgauge;
	complex *delta;
	complex *gx;
	GaugeFixArg(gauge pgaugein, gauge DELTA, gauge GX, const int XX[4]){
		for(int dir=0; dir<4; ++dir) X[dir] = XX[dir];
		threads = X[0]*X[1]*X[2]*X[3];
		pgauge = pgaugein.GetPtr();
		delta = DELTA.GetPtr();
		gx = GX.GetPtr();
	}
	GaugeFixArg(gauge pgaugein, gauge DELTA, complex *gx, const int XX[4]):gx(gx){
		for(int dir=0; dir<4; ++dir) X[dir] = XX[dir];
		threads = X[0]*X[1]*X[2]*X[3];
		pgauge = pgaugein.GetPtr();
		delta = DELTA.GetPtr();
	}
};







template <typename Real> 
__global__ void kernel_gauge_set_invpsq(GaugeFixArg<Real> arg, Real *invpsq){
	int id = INDEX1D();
	if(id >= arg.threads) return;
	int x1 = id/(arg.X[2] * arg.X[3] * arg.X[0]);
	int x0 = (id/(arg.X[2] * arg.X[3])) % arg.X[0];
	int x3 = (id/arg.X[2]) % arg.X[3];
	int x2 = id % arg.X[2];
	//id  =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2]; 
	Real sx = sin( (Real)x0 * FL_UNITARIZE_PI / (Real)arg.X[0]);
	Real sy = sin( (Real)x1 * FL_UNITARIZE_PI / (Real)arg.X[1]);
	Real sz = sin( (Real)x2 * FL_UNITARIZE_PI / (Real)arg.X[2]);
	Real st = sin( (Real)x3 * FL_UNITARIZE_PI / (Real)arg.X[3]);
	Real sinsq = sx * sx + sy * sy + sz * sz + st * st;
	Real prcfact = 0.0;
	//The FFT normalization is done here
	if ( sinsq > 0.00001 )   prcfact = 4.0 / (sinsq * (Real)arg.threads);   
	invpsq[id] = prcfact;
}



template<typename Real>
class GaugeFixSETINVPSP : Tunable {
	GaugeFixArg<Real> arg;
	Real *invpsq;
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFixSETINVPSP(GaugeFixArg<Real> &arg, Real *invpsq) : arg(arg), invpsq(invpsq) {    }
	~GaugeFixSETINVPSP () { }

	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_set_invpsq<Real><<<tp.grid, tp.block, 0, stream>>>(arg, invpsq);
	}

	TuneKey tuneKey() const {
		std::stringstream vol, aux;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		aux << "threads=" <<arg.threads << ",prec=" << sizeof(Real);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), "none", aux.str().c_str());

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){} 

	long long flop() const { return 21LL * arg.threads;}

	long long bytes() const {return sizeof(Real) * arg.threads;}
}; 



template<typename Real>
__global__ void kernel_gauge_mult_norm_2D(complex *data, Real *invpsq, int size){
	int id = INDEX1D();
	if(id < size) data[id] = data[id] * invpsq[id]; 
}


template<typename Real>
class GaugeFixINVPSP : Tunable {
	GaugeFixArg<Real> arg;
	complex *data;
	Real *invpsq;
	complex *tmp;
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }
	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_gauge_mult_norm_2D<Real><<<tp.grid, tp.block, 0, stream>>>(data, invpsq, arg.threads);
	}
public:
	GaugeFixINVPSP(GaugeFixArg<Real> &arg): arg(arg){ 
		invpsq = (Real*) dev_malloc(sizeof(Real) * arg.threads);
		GaugeFixSETINVPSP<Real> setinvpsp(arg, invpsq);
		setinvpsp.apply(0);
	}
	~GaugeFixINVPSP () {dev_free(invpsq);}
	void Run( complex *datain, const cudaStream_t &stream){
		data = datain;
		apply(stream);
	}
	TuneKey tuneKey() const {
		std::stringstream vol, aux;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		aux << "threads=" <<arg.threads << ",prec=" << sizeof(Real);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), "none", aux.str().c_str());

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){
		//since delta contents are irrelevant at this point, we can swap gx with delta
		tmp = (complex*)dev_malloc(sizeof(complex)*arg.threads);
		CUDA_SAFE_CALL(cudaMemcpy(tmp, data, sizeof(complex)*arg.threads, cudaMemcpyDeviceToDevice));
	}
	void postTune(){
		CUDA_SAFE_CALL(cudaMemcpy(data, tmp, sizeof(complex)*arg.threads, cudaMemcpyDeviceToDevice));
		dev_free(tmp);
	}
	long long flop() const { return 2LL * arg.threads; }
	long long bytes() const { return 5LL * sizeof(Real) * arg.threads; } 
}; 









/**
	@brief Calculate g(x). g(x) is written in even/odd lattice sites separately
	@param gx complex array to store g(x)
	@param Delta complex array with IFF alpha/2 (pmax^2a^2)/(p^2a^2)FFT(...)
	@param half_alpha alpha/2
template <bool UseTex, ArrayType atypedelta, ArrayType atypegx, class Real> 
__global__ void kernel_gauge_SUM_REUNIT_GXEO(complex *gx, complex *Delta, Real half_alpha)
*/
template <bool UseTex, ArrayType atypedelta, ArrayType atypegx, class Real> 
__global__ void kernel_gauge_SUM_REUNIT_GXEO(GaugeFixArg<Real> arg, Real half_alpha){
	int id     = INDEX1D();
	if(id >= arg.threads) return;
	msun de = DELTA_LOAD<UseTex, atypedelta, Real>( arg.delta, id);//, arg.threads);		
	msun g = msun::unit();
	g += de * half_alpha;
	reunit_link<Real>( &g );
	GAUGE_SAVE<atypegx, Real>(arg.gx, g, id, arg.threads);
}








template <bool UseTex, ArrayType atypedelta, ArrayType atypegx, class Real> 
class GaugeFix_GX : Tunable {
	GaugeFixArg<Real> arg;
	gauge gx;
	gauge delta;
	Real half_alpha;
	typedef void (*TFuncPtr)(GaugeFixArg<Real>, Real);
    TFuncPtr kernel_pointer;
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:

	GaugeFix_GX(GaugeFixArg<Real> &arg, Real alpha, gauge gx, gauge delta) : arg(arg), gx(gx), delta(delta) {  
	half_alpha = alpha * 0.5; 
	if(gx.EvenOdd()) errorCULQCD("gx not set in even/odd format...");
	if(delta.EvenOdd()) errorCULQCD("delta cannot be in even/odd format...");
	kernel_pointer = NULL;
    kernel_pointer = &kernel_gauge_SUM_REUNIT_GXEO<UseTex, atypedelta, atypegx, Real>;		
	if(kernel_pointer == NULL) errorCULQCD("No kernel kernel_gauge_SUM_REUNIT_GXEO function exist for this gauge array...");
	}
	~GaugeFix_GX () { }

	void setAlpha(Real alpha){ half_alpha = alpha * 0.5; }


	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_pointer<<<tp.grid, tp.block, 0, stream>>>(arg, half_alpha);
	}

	TuneKey tuneKey() const {
		std::stringstream vol, aux;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		aux << "threads=" <<arg.threads << ",prec=" << sizeof(Real);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), gx.ToStringArrayType().c_str(), aux.str().c_str());

	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){}
	long long flop() const { 
	#if (NCOLORS == 3)
		unsigned int ThreadFlop = 126;
	#else
		unsigned int tmp_gs = 0;
		unsigned int tmp_det = 0;
		for(int i = 0; i<NCOLORS;i++){
	        tmp_gs+=i+1;
	        tmp_det+=i;
		}
		tmp_det = tmp_gs * NCOLORS * 8 + tmp_det * (NCOLORS * 8 + 11);
		tmp_gs = tmp_gs * NCOLORS * 16 + NCOLORS * (NCOLORS * 6 + 2);
		unsigned int ThreadFlop = tmp_gs + tmp_det;
	#endif
		return ( 4LL * NCOLORS * NCOLORS + ThreadFlop + delta.getNumFlop(true) + gx.getNumFlop(false) ) * arg.threads;
	}
  	long long bytes() const { return (gx.getNumParams() + delta.getNumParams()) * sizeof(Real) * arg.threads; }

}; 






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
	@brief Apply g(x) do gauge fixing array. kernel for g(x) and gauge array written in even/odd lattice sites separately
	@param array gauge array to be fixed.
	@param gx complex array with g(x).
*/
template <bool UseTex, ArrayType atype, ArrayType atypegx, class Real> 
__global__ void kernel_gauge_fix_U( complex *array, complex *gx){
	int idd     = INDEX1D();
	if(idd < DEVPARAMS::Volume){
		msun g = GX_LOAD<UseTex, atypegx, Real>(gx, idd, DEVPARAMS::Volume);
		int x[4];
		getCoords4(x, idd, DEVPARAMS::Grid);		
		for(int nu = 0; nu < 4; nu++){
			msun U = GAUGE_LOAD<UseTex, atype, Real>( array, idd + nu * DEVPARAMS::Volume, DEVPARAMS::Volume * 4);	
			msun U_temp = g * U;
			msun g0 = (GX_LOAD_DAGGER<UseTex, atypegx, Real>(gx,  neighborIndexPlusOne(x, nu, DEVPARAMS::Grid), DEVPARAMS::Volume));
			U = U_temp * g0;
			GAUGE_SAVE<atype, Real>(array, U, idd + nu * DEVPARAMS::Volume, DEVPARAMS::Volume * 4);		
		}
	}
}





template<bool UseTex, ArrayType atype, ArrayType atypegx, class Real>
class GaugeFix : Tunable {
	GaugeFixArg<Real> arg;
	gauge &pgauge;
	gauge gx;
	typedef void (*TFuncPtr)(complex *, complex *);
    TFuncPtr kernel_pointer;
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFix(gauge &pgauge, gauge GX, GaugeFixArg<Real> &arg):pgauge(pgauge),arg(arg), gx(GX){ 

	if(pgauge.EvenOdd() ) errorCULQCD("gauge cannot be in even/odd format...");
	if(GX.EvenOdd()) errorCULQCD("gx not set in even/odd format...");
	kernel_pointer = NULL;
    kernel_pointer = &kernel_gauge_fix_U<UseTex, atype, atypegx, Real>;	
	if(kernel_pointer == NULL) errorCULQCD("No kernel GaugeFix function exist for this gauge array...");
   }
	~GaugeFix () { }


	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		kernel_pointer<<<tp.grid, tp.block, 0, stream>>>(pgauge.GetPtr(), arg.gx);
	}

	TuneKey tuneKey() const {
		std::stringstream vol, aux;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		aux << "threads=" <<arg.threads << ",prec=" << sizeof(Real);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), pgauge.ToStringArrayType().c_str(), aux.str().c_str());
	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	//need this
	void preTune() { pgauge.Backup(); }
	void postTune() { pgauge.Restore(); }
	long long flop() const { 
		return (16LL * NCOLORS * NCOLORS * (NCOLORS * 5 - 2) + 5LL * gx.getNumFlop(true) + 4LL * pgauge.getNumFlop(true) + 4LL * pgauge.getNumFlop(false)) * arg.threads;
	}
  	long long bytes() const { return (8LL * pgauge.getNumParams() + 5LL * gx.getNumParams()) * sizeof(Real) * arg.threads;}  

}; 






template <bool UseTex, ArrayType atype, ArrayType atypedelta, class Real> 
__global__ void kernel_gauge_fix_U_NEW( complex *array, complex *delta, Real half_alpha){
	int idd     = INDEX1D();
	if(idd >= DEVPARAMS::Volume) return;

	msun de = DELTA_LOAD<UseTex, atypedelta, Real>( delta, idd);//, arg.threads);		
	msun g = msun::unit();
	g += de * half_alpha;
	reunit_link<Real>( &g );
	int x[4];
	getCoords4(x, idd, DEVPARAMS::Grid);
	for(int nu = 0; nu < 4; nu++){
		msun U = GAUGE_LOAD<UseTex, atype, Real>( array, idd + nu * DEVPARAMS::Volume, DEVPARAMS::Volume * 4);	
		msun U_temp = g * U;
		de = DELTA_LOAD<UseTex, atypedelta, Real>(delta,  neighborIndexPlusOne(x, nu, DEVPARAMS::Grid));	
		msun g0 = msun::unit();
		g0 += de * half_alpha;
		reunit_link<Real>( &g0 );
		U = U_temp * g0.dagger();
		GAUGE_SAVE<atype, Real>(array, U, idd + nu * DEVPARAMS::Volume, DEVPARAMS::Volume * 4);		
	}
}

template<bool UseTex, ArrayType atype, ArrayType atypedelta, class Real>
class GaugeFixNEW : Tunable {
	GaugeFixArg<Real> arg;
	gauge &pgauge;
	gauge delta;
	Real half_alpha;
	typedef void (*TFuncPtr)(complex *, complex *, Real);
    TFuncPtr kernel_pointer;
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

public:
	GaugeFixNEW(gauge &pgauge, gauge &delta, Real alpha, GaugeFixArg<Real> &arg):pgauge(pgauge),arg(arg), delta(delta){ 
	half_alpha = alpha * 0.5;
	if(pgauge.EvenOdd()) errorCULQCD("gauge cannot be in even/odd format...");
	if(delta.EvenOdd()) errorCULQCD("delta cannot be in even/odd format...");
	kernel_pointer = NULL;
    kernel_pointer = &kernel_gauge_fix_U_NEW<UseTex, atype, atypedelta, Real>;		
	if(kernel_pointer == NULL) errorCULQCD("No kernel kernel_gauge_fix_U_NEW function exist for this gauge array...");
   }
	~GaugeFixNEW () { }

	void setAlpha(Real alpha){ half_alpha = alpha * 0.5; }


	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		//kernel_pointer<<<tp.grid, tp.block, 0, stream>>>(pgauge.GetPtr(), delta.GetPtr(), half_alpha);
		kernel_pointer<<<tp.grid, tp.block, 0, stream>>>(pgauge.GetPtr(), arg.delta, half_alpha);
	}

	TuneKey tuneKey() const {
		std::stringstream vol, aux;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		aux << "threads=" <<arg.threads << ",prec=" << sizeof(Real);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), pgauge.ToStringArrayType().c_str(), aux.str().c_str());
	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	//need this
	void preTune() { pgauge.Backup(); }
	void postTune() { pgauge.Restore(); }

	long long flop() const { 
		#if (NCOLORS == 3)
		unsigned int ThreadFlop = 126;
		#else
		unsigned int tmp_gs = 0;
		unsigned int tmp_det = 0;
		for(int i = 0; i<NCOLORS;i++){
	        tmp_gs+=i+1;
	        tmp_det+=i;
		}
		tmp_det = tmp_gs * NCOLORS * 8 + tmp_det * (NCOLORS * 8 + 11);
		tmp_gs = tmp_gs * NCOLORS * 16 + NCOLORS * (NCOLORS * 6 + 2);
		unsigned int ThreadFlop = tmp_gs + tmp_det;
		#endif
		return (5LL * delta.getNumFlop(true) + 4LL * pgauge.getNumFlop(true) + 4LL * pgauge.getNumFlop(false) + \
			20LL * NCOLORS * NCOLORS + 5LL * ThreadFlop + 16LL * NCOLORS * NCOLORS * (NCOLORS * 5 - 2) ) * arg.threads;
	}
  	long long bytes() const { return (8LL * pgauge.getNumParams() + 5LL * delta.getNumParams()) * sizeof(Real) * arg.threads;} 
}; 




template <typename Real>
struct GaugeFixFFTRotateArg {
	int threads; // number of active threads required
	int X[4]; // grid dimensions
	complex *in;
	complex *out;
	GaugeFixFFTRotateArg(const int XX[4]){
		for(int dir=0; dir<4; ++dir) X[dir] = XX[dir];
		threads = X[0]*X[1]*X[2]*X[3];
	}
};





template <int tile_dim, int block_rows, int direction, typename Real> 
__global__ void fft_rotate_kernel_2D2D_Optimized(GaugeFixFFTRotateArg<Real> arg){
	complex *tile = SharedMemory<complex>();
    unsigned int x = blockIdx.x * tile_dim + threadIdx.x;
    unsigned int yBase = blockIdx.y * tile_dim + threadIdx.y;
    unsigned int Sx;
    unsigned int Sy;
	if(direction == 0){
		Sx = arg.X[0]* arg.X[1];
		Sy = arg.X[2] * arg.X[3];
	}
	if(direction == 1){
		Sy = arg.X[0]*arg.X[1];
		Sx = arg.X[2] * arg.X[3];
	}
    if(x < Sx) {
        for(unsigned int j = 0; j < tile_dim; j+= block_rows) {
            unsigned int y = yBase + j;
            if(y >= Sy) break;
            tile[threadIdx.y + j + threadIdx.x * (tile_dim+1)] = arg.in[y * Sx + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * tile_dim + threadIdx.x;
    yBase = blockIdx.x * tile_dim + threadIdx.y;
    if(x < Sy) {
        for(unsigned int j = 0; j < tile_dim; j += block_rows) {
            unsigned int y = yBase + j;
            if(y >= Sx) break;
            arg.out[y*Sy + x] = tile[threadIdx.x + (threadIdx.y + j) * (tile_dim+1)];
        }  
    }
}


#define LAUNCH_KERNEL_ROTATE(kernel, blockdimX, tp, stream, arg, ...)     \
  switch (tp.block.y) {             \
  case 8:            \
	    kernel<blockdimX, 8,__VA_ARGS__><<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);   \
	    break;                \
  case 16:  \
	    kernel<blockdimX, 16,__VA_ARGS__><<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);   \
	    break;                \
  default:                \
    errorCULQCD("%s not implemented for %d:%d threads\n", #kernel, tp.block.x, tp.block.y); \
    }
       


template<int dir, typename Real>
class GaugeFixFFTRotate : Tunable {
	GaugeFixFFTRotateArg<Real> arg;
private:
	unsigned int sharedBytesPerThread() const { return 0; }
	unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
	bool tuneSharedBytes() const { return false; } // Don't tune shared memory
	bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
	unsigned int minThreads() const { return arg.threads; }

  dim3 createGrid   (const dim3 &block) const { 
  	dim3 grid;
  	if(dir==0){
  		dim3 grid0((PARAMS::Grid[0]*PARAMS::Grid[1] + block.x - 1)/block.x, (PARAMS::Grid[2]*PARAMS::Grid[3] + block.y - 1)/block.y);
  		grid = grid0;
  	}
	if(dir==1){
		dim3 grid1((PARAMS::Grid[2]*PARAMS::Grid[3] + block.x - 1)/block.x, (PARAMS::Grid[0]*PARAMS::Grid[1] + block.y - 1)/block.y);
  		grid = grid1;
  	}
	return grid;
}
bool advanceBlockDim  (TuneParam &param) const {
	const int blocky = 8;
	param.block.x = 32;
	param.block.y += blocky;
    if(param.block.y < param.block.x && 2*param.block.y <= param.block.x){
    	param.shared_bytes = (param.block.x+1)*(param.block.x)*sizeof(complex);
    	param.grid = createGrid(param.block);
    	return true;
    }
    else return false;
}
  virtual void initTuneParam(TuneParam &param) const{ 
    dim3 block(32, 8);
	dim3 grid0((PARAMS::Grid[0]*PARAMS::Grid[1] + block.x - 1)/block.x, (PARAMS::Grid[2]*PARAMS::Grid[3] + block.y - 1)/block.y);
	dim3 grid1((PARAMS::Grid[2]*PARAMS::Grid[3] + block.x - 1)/block.x, (PARAMS::Grid[0]*PARAMS::Grid[1] + block.y - 1)/block.y);
	param.block = block;
	if(dir==0) param.grid = grid0;
	else param.grid = grid1;
	param.shared_bytes = (param.block.x+1)*(param.block.x)*sizeof(complex);
	}

  /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const{ 
    dim3 block(32, 8);
	dim3 grid0((PARAMS::Grid[0]*PARAMS::Grid[1] + block.x - 1)/block.x, (PARAMS::Grid[2]*PARAMS::Grid[3] + block.y - 1)/block.y);
	dim3 grid1((PARAMS::Grid[2]*PARAMS::Grid[3] + block.x - 1)/block.x, (PARAMS::Grid[0]*PARAMS::Grid[1] + block.y - 1)/block.y);
	param.block = block;
	if(dir==0) param.grid = grid0;
	else param.grid = grid1;
	param.shared_bytes = (param.block.x+1)*(param.block.x)*sizeof(complex);
	}

	void apply(const cudaStream_t &stream){
		TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
		LAUNCH_KERNEL_ROTATE(fft_rotate_kernel_2D2D_Optimized, 32, tp, stream, arg, dir, Real);
	}
public:
	GaugeFixFFTRotate(GaugeFixFFTRotateArg<Real> &arg)
: arg(arg) {
	}
	~GaugeFixFFTRotate () {}
	void Run(complex *data_in, complex *data_out, const cudaStream_t &stream){
		arg.in = data_in;
		arg.out = data_out;
		apply(stream);
	}
	TuneKey tuneKey() const {
		std::stringstream vol, aux;
		vol << arg.X[0] << "x";
		vol << arg.X[1] << "x";
		vol << arg.X[2] << "x";
		vol << arg.X[3];
		aux << "threads=" <<arg.threads << ",prec=" << sizeof(Real);
		return TuneKey(vol.str().c_str(), typeid(*this).name(), "none", aux.str().c_str());
	}
	std::string paramString(const TuneParam &param) const {
		std::stringstream ps;
		ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
		ps << "shared=" << param.shared_bytes;
		return ps.str();
	}
	void preTune(){}
	void postTune(){}
	long long flop() const { return 0; }
	long long bytes() const { return 2LL * sizeof(complex) * arg.threads; } 
}; 













///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, ArrayType atype, ArrayType atypeDeltax, ArrayType atypeGx, class Real, int DIR> 
complex CALL_FFT(gauge _pgauge, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, bool useGx){

	Timer gfltime;
	gfltime.start();

	gauge _delta(atypeDeltax, Device, PARAMS::Volume, false);
	gauge _gx(atypeGx, Device, _pgauge.EvenOdd());
	complex *_GxPtr;	
	if(useGx){
		_gx.Allocate(PARAMS::Volume);
		_GxPtr = _gx.GetPtr();
		printfCULQCD("Calculating G(x) array...\n");
	}
	else{
		_GxPtr = (complex*)dev_malloc(sizeof(complex)*PARAMS::Volume);
		printfCULQCD("Not calculating G(x) array...\n");
	}
	printfCULQCD("Using 2D FFTs + 2D FFTs with cufftPlanMany using %d/%d elements...\n", _delta.getNumElems(), NCOLORS * NCOLORS);
	if(useGx){
		long long sizeused = _pgauge.Bytes() + _delta.Bytes() + _gx.Bytes() + PARAMS::Volume*sizeof(Real);
		COUT << "Device memory used not accounting CUFFT: " << sizeused/(1024*1024) << " MB" << endl;
	}
	else{
		long long sizeused = _pgauge.Bytes() + _delta.Bytes() + PARAMS::Volume*(sizeof(complex)+sizeof(Real));
		COUT << "Device memory used not accounting CUFFT: " << sizeused/(1024*1024) << " MB" << endl;
	}

	//------------------------------------------------------------------------
	// Bind TEXTURES_FFT if PARAMS::UseTex is True
	//------------------------------------------------------------------------ 
	GAUGE_TEXTURE(_pgauge.GetPtr(), true);
	DELTA_TEXTURE(_delta.GetPtr(), true);
	if(useGx) GX_TEXTURE(_gx.GetPtr(), true);
	//------------------------------------------------------------------------
	//------------------------------------------------------------------------
	// Create two 2D FFT plans.
	//------------------------------------------------------------------------
	int4 size = make_int4( PARAMS::Grid[0], PARAMS::Grid[1], PARAMS::Grid[2], PARAMS::Grid[3] );
	cufftHandle plan_xy;
	cufftHandle plan_zt;
	SetPlanFFT2DMany( plan_zt, size, 0, _pgauge.GetPtr()); //for space and time ZT
	SetPlanFFT2DMany( plan_xy, size, 1, _pgauge.GetPtr());//with space only XY

	//------------------------------------------------------------------------
	COUT << "................................................." << endl;
	complex data; //.real()-> Fg :: .imag() -> Theta
	Real theta_new = 0.0;
	Real Fg_new = 0.0;
	Real Fg_old = 0.0;

	GaugeFixArg<Real> arg(_pgauge, _delta, _GxPtr, PARAMS::Grid);
	GaugeFixFFTRotateArg<Real> arg_rotate(PARAMS::Grid);
	GaugeFixFFTRotate<0, Real> GFRotate0(arg_rotate);
	GaugeFixFFTRotate<1, Real> GFRotate1(arg_rotate);
	//------------------------------------------------------------------------
	// Precalculate pmax^2/p^2, also includes the FFT normalization
	//------------------------------------------------------------------------
	GaugeFixINVPSP<Real> invpsp(arg);
	GaugeFixFFTQuality<UseTex, atype, atypeDeltax, Real, DIR> quality(_pgauge, _delta, _GxPtr, PARAMS::Grid);
	//if useGx=true
	GaugeFix_GX<UseTex, atypeDeltax, atypeGx, Real> calcGX(arg, alpha, _gx, _delta);
	GaugeFix<UseTex, atype, atypeGx, Real> gfix(_pgauge, _gx, arg);
	//if useGx=false
	GaugeFixNEW<UseTex, atype, atypeDeltax, Real> gfixNEW(_pgauge, _delta, alpha, arg);
	//end
	//------------------------------------------------------------------------
	// Measure initial gauge quality and calculate Delta(x)
	//------------------------------------------------------------------------
	data = quality.Run();
	theta_new = data.imag();
	Fg_new =  data.real();
	printfCULQCD("Iter: %d\tFg = %.12e\ttheta = %.12e\n",0, Fg_new, theta_new );
	Fg_old = Fg_new;
	//------------------------------------------------------------------------
	// Do gauge fix
	//------------------------------------------------------------------------
	Real diff = 0.0;
	int iterations = 0;
	for(int iter = 1; iter <= maxsteps; ++iter){
		iterations++;
		//------------------------------------------------------------------------
		// Perform FFT to each SU(3) element of Delta (x)
		//------------------------------------------------------------------------
		for(int k = 0; k < _delta.getNumElems(); k++){	
			// Set a pointer do the element k in lattice volume
			// each element is stored with stride lattice volume
			// it uses gx as temporary array!!!!!!
			//------------------------------------------------------------------------
			complex *_array = _delta.GetPtr() + k * _delta.Size();
			//////  2D FFT + 2D FFT
			//------------------------------------------------------------------------
			// Perform FFT on xy plane
			//------------------------------------------------------------------------
			ApplyFFT(plan_xy, _array, arg.gx, CUFFT_FORWARD);
			//------------------------------------------------------------------------
			// Rotate hypercube, xyzt -> ztxy
			//------------------------------------------------------------------------
			GFRotate0.Run(arg.gx, _array, 0);
			//------------------------------------------------------------------------
			// Perform FFT on zt plane
			//------------------------------------------------------------------------      
			ApplyFFT(plan_zt, _array, arg.gx, CUFFT_FORWARD);
			//------------------------------------------------------------------------
			// Normalize FFT and apply pmax^2/p^2
			//------------------------------------------------------------------------
			invpsp.Run(arg.gx, 0);
			//------------------------------------------------------------------------
			// Perform IFFT on zt plane
			//------------------------------------------------------------------------  
			ApplyFFT(plan_zt, arg.gx, _array, CUFFT_INVERSE);
			//------------------------------------------------------------------------
			// Rotate hypercube, ztxy -> xyzt
			//------------------------------------------------------------------------
			GFRotate1.Run(_array, arg.gx, 0);
			//------------------------------------------------------------------------
			// Perform IFFT on xy plane
			//------------------------------------------------------------------------    
			ApplyFFT(plan_xy, arg.gx, _array, CUFFT_INVERSE);  
		}
		if(useGx) {
			//------------------------------------------------------------------------
			// Calculate g(x)
			//------------------------------------------------------------------------
			calcGX.apply(0);
			//------------------------------------------------------------------------
			// Apply gauge fix to current gauge field
			//------------------------------------------------------------------------
			gfix.apply(0);
		}
		else{
			//------------------------------------------------------------------------
			// Apply gauge fix to current gauge field
			//------------------------------------------------------------------------
			gfixNEW.apply(0);
		}
		//------------------------------------------------------------------------
		// Measure gauge quality and recalculate new Delta(x)
		//------------------------------------------------------------------------
		data = quality.Run();
		theta_new = data.imag();
		Fg_new =  data.real();
		diff = abs(Fg_old - Fg_new);
		if((iter%verbose)==0) printfCULQCD("Iter: %d\tFg = %.12e\ttheta = %.12e\tDelta = %.12e\n", iter, Fg_new, theta_new, diff );
		if ( landautune && ((Fg_new - Fg_old) < -1e-14) ) {
			if(alpha > 0.01){
				alpha=0.95 * alpha;
				if(useGx) calcGX.setAlpha(alpha);
				else gfixNEW.setAlpha(alpha);
				printfCULQCD(">>>>>>>>>>>>>> Warning: changing alpha down -> %.4e\n",	alpha );
			}
		}
		Fg_old = Fg_new; 
		//------------------------------------------------------------------------
		// Check gauge fix quality criterium
		//------------------------------------------------------------------------
		#ifdef USE_THETA_STOP_GAUGEFIX
		if( theta_new < stopvalue ) break;
		#else
		if( diff < stopvalue ) break;
		#endif
	}
	if((iterations%verbose)!=0) printfCULQCD("Iter: %d\tFg = %.12e\ttheta = %.12e\tDelta = %.12e\n", iterations, Fg_new, theta_new, diff );
	string my_name ="";
	if(DIR==4) my_name = "Landau gauge fixing";
	else if(DIR==3) my_name =  "Coulomb gauge fixing";
	COUT << "Finishing " << my_name << " using 2D FFTs + 2D FFTs..." << endl;
	//------------------------------------------------------------------------
	// Destroy CUFFT plans.
	//------------------------------------------------------------------------
	CUFFT_SAFE_CALL(cufftDestroy(plan_zt));
	CUFFT_SAFE_CALL(cufftDestroy(plan_xy));	
	//------------------------------------------------------------------------
	// Unbind TEXTURES_FFT if used
	//------------------------------------------------------------------------
	GAUGE_TEXTURE(_pgauge.GetPtr(), false);
	DELTA_TEXTURE(_delta.GetPtr(), false);
	if(useGx) GX_TEXTURE(_gx.GetPtr(), false);
	//------------------------------------------------------------------------
	// Release all temporary arrays
	//------------------------------------------------------------------------
	gfltime.stop();
	#ifndef TIMMINGS
	COUT << my_name << " with FFTs -> Time: " << gfltime.getElapsedTimeInSec() << " s" << endl;
	#endif

#ifdef TIMMINGS
	float flops = 21.0 * PARAMS::Volume + (float)quality.flop() * (iterations + 1);
	float bytes = (float)PARAMS::Volume * sizeof(Real) + quality.bytes() * (iterations + 1);
	flops += (GFRotate0.flop() + GFRotate1.flop() + invpsp.flop()) * (iterations * _delta.getNumElems());
	bytes += (GFRotate0.bytes() + GFRotate1.bytes() + invpsp.bytes())* (iterations * _delta.getNumElems());
	if(useGx) {
		flops += (calcGX.flop() + gfix.flop()) * iterations;
		bytes += (calcGX.bytes() + gfix.bytes()) * iterations;
	}
	else{
		flops += gfixNEW.flop() * iterations;
		bytes += gfixNEW.bytes() * iterations;
	}
	float fftflop = 10.0 * (log2((float)( PARAMS::Grid[0] * PARAMS::Grid[1]) ) + log2( (float)(PARAMS::Grid[2] * PARAMS::Grid[3] )));
	fftflop *= ( (float)iterations * _delta.getNumElems() * (float)PARAMS::Volume);
	//Not accounting Bytes read/write in cuFFT
	float TotalGBytes = bytes / (gfltime.getElapsedTimeInSec() * (float)(1 << 30));
	float TotalGFlops = ((flops + fftflop) * 1.0e-9) / gfltime.getElapsedTimeInSec();
	COUT << my_name << ":  " <<  gfltime.getElapsedTimeInSec() << " s\t"  << TotalGBytes << " GB/s\t" << TotalGFlops << " GFlops"  << endl;
#endif
	_delta.Release();
	//_gx.GetPtr()/_GxPtr and  array is release inside  GaugeFixFFTQuality...
	COUT << "###############################################################################" << endl;
	return complex::make_complex(theta_new,(Real)iterations);
}

template <ArrayType atype, ArrayType atypeDeltax, ArrayType atypeGx, class Real, int DIR> 
complex CALL_FFT_TEX(gauge _pgauge, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, bool useGx){
	if(PARAMS::UseTex)
		return CALL_FFT<true, atype, atypeDeltax, atypeGx, Real, DIR>(_pgauge, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
	else
		return CALL_FFT<false, atype, atypeDeltax, atypeGx, Real, DIR>(_pgauge, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
}
template <ArrayType atype, ArrayType atypeDeltax, ArrayType atypeGx, class Real> 
complex CALL_FFT_DIR(gauge _pgauge, int DIR, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, bool useGx){
	if(DIR==3)
		return CALL_FFT_TEX<atype, atypeDeltax, atypeGx, Real, 3>(_pgauge, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
	else
		return CALL_FFT_TEX<atype, atypeDeltax, atypeGx, Real, 4>(_pgauge, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
}
template <ArrayType atype, ArrayType atypeDeltax, class Real> 
complex CALL_FFT_GX(gauge _pgauge, int DIR, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, bool useGx, ArrayType atypeGx){
	#if (NCOLORS == 3)
    if(atypeGx == SOA) return CALL_FFT_DIR<atype, atypeDeltax, SOA, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
    else if(atypeGx == SOA12) return CALL_FFT_DIR<atype, atypeDeltax, SOA12, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
    else if(atypeGx == SOA8) return CALL_FFT_DIR<atype, atypeDeltax, SOA8, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx);
    else errorCULQCD("Not defined...\n");
    return complex::make_complex(9999.,999999999999.);
	#else
    return CALL_FFT_DIR<atype, atypeDeltax, SOA, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx);	
    #endif
}
template <ArrayType atype, class Real> 
complex CALL_FFT_DELTA(gauge _pgauge, int DIR, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, ArrayType atypeDeltax, bool useGx, ArrayType atypeGx){
	#if (NCOLORS == 3)
    if(atypeDeltax == SOA) return CALL_FFT_GX<atype, SOA, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx, atypeGx);
    else if(atypeDeltax == SOA12A) return CALL_FFT_GX<atype, SOA12A, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx, atypeGx);
    else errorCULQCD("Not defined for SOA12 and SOA8...\n");
    return complex::make_complex(9999.,999999999999.);
	#else
    return CALL_FFT_GX<atype, SOA, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, useGx, atypeGx);	
    #endif
}
template <class Real> 
complex CALL_FFT_PGAUGE(gauge _pgauge, int DIR, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, ArrayType atypeDeltax, bool useGx, ArrayType atypeGx){
	#if (NCOLORS == 3)
    if(_pgauge.Type() == SOA) return CALL_FFT_DELTA<SOA, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, atypeDeltax, useGx, atypeGx);
    else if(_pgauge.Type() == SOA12) return CALL_FFT_DELTA<SOA12, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, atypeDeltax, useGx, atypeGx);
    else if(_pgauge.Type() == SOA8) return CALL_FFT_DELTA<SOA8, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, atypeDeltax, useGx, atypeGx);
    else errorCULQCD("Not defined...\n");
    return complex::make_complex(9999.,999999999999.);
	#else
    return CALL_FFT_DELTA<SOA, Real>(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, atypeDeltax, useGx, atypeGx);	
    #endif
}



/**
   @brief Apply Landau/Coulomb Gauge Fixing with Steepest Descent Method with Fourier Acceleration
   @param array gauge field to be fixed
   @param DIR DIR=4 for Landau gauge Fixing and DIR=3 for Coulomb gauge fixing
   @param alpha constant for the method, optimal value is 0.08
   @param landautune if true auto tune method
   @param stopvalue criterium to stop the method, precision
   @param maxsteps maximum number of iterations
*/ 
template <class Real> 
complex GaugeFixingFFT(gauge _pgauge, int DIR, Real alpha, bool landautune, Real stopvalue, int maxsteps, int verbose, ArrayType atypeDeltax, bool useGx, ArrayType atypeGx){ 
	#ifdef MULTI_GPU
	if(numnodes() > 1 ){
		COUT << "NOT IMPLEMENTED YET FOR MULTI GPUs!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		return complex::make_complex(9999.,999999999999.);
	}
	#endif
	COUT << "###############################################################################" << endl;
	string my_name ="";
	if(DIR==4) my_name = "Landau gauge fixing";
	else if(DIR==3) my_name =  "Coulomb gauge fixing";
	else{
        COUT << "DIR can only be 3, for Coulomb, or 4, for Landau.\nNo gauge fixing applied." << endl;
        return complex::make_complex(9999.,999999999999.);
	}
    if(_pgauge.EvenOdd()){
		COUT << "Only implemented for normal order arrays.\nNo gauge fixing applied." << endl;
		return complex::make_complex(9999.,999999999999.);
	}
	COUT << "Applying " << my_name << "." << endl;
	COUT << "\tAlpha parameter of the Steepest Descent Method: " << alpha << endl;
	COUT << "\tAuto tune active: " << ( landautune ? "yes" : "no") << endl;
	COUT << "\tStop criterium: " << stopvalue << endl;
	COUT << "\tMaximum number of iterations: " << maxsteps << endl;
	if(verbose < 1) verbose = maxsteps;
	COUT << "\tPrint convergence results at every " << verbose << " steps" << endl;
	return CALL_FFT_PGAUGE(_pgauge, DIR, alpha, landautune, stopvalue, maxsteps, verbose, atypeDeltax, useGx, atypeGx);
}	
template complexs
GaugeFixingFFT<float>(gauges _pgauge, int DIR, float alpha, bool landautune, float stopvalue, int maxsteps, int verbose, ArrayType atypeDeltax, bool useGx, ArrayType atypeGx);
template complexd
GaugeFixingFFT<double>(gauged _pgauge, int DIR, double alpha, bool landautune, double stopvalue, int maxsteps, int verbose, ArrayType atypeDeltax, bool useGx, ArrayType atypeGx);





}

}
