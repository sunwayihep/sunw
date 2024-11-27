

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>


#include <device_load_save.h>
#include <cuda_common.h>
#include <constants.h>
#include <index.h>
#include <reduction.h>
#include <timer.h>
#include <texture_host.h>
#include <comm_mpi.h>


#include <tune.h>

#include <projectlink.h>
#include <reunitlink.h>


#include <eesu3.h>


using namespace std;


namespace CULQCD{



/*--------------------------------------------------------------------*/
/* Construct Q from smeared link V and unsmeared link U */
/* Quick and dirty code - can be optimized for SU(3) */

/* a = traceless-hermitian part of b.  b and a may be equivalent. */
template<class Real>
__host__ __device__ inline void traceless_hermitian_sun(msun &a, msun b){
  a = b + b.dagger();
  complex t = a.trace() / (Real)NCOLORS;  
#if (NCOLORS == 3) && defined(SU3FASTER)
  a.e00 -= t;
  a.e11 -= t;
  a.e22 -= t;
#else
    for(int i=0;i<NCOLORS;i++)
	a.e[i][i] -= t;
#endif
  a *= 0.5;
}

/*--------------------------------------------------------------------*/

/*
* return Tr( A*B )   						*
*/
template<class Real>
__host__ __device__ inline complex complextrace_sun_nn( msun a, msun b ) {

  complex sum = complex::zero();
#if (NCOLORS == 3) && defined(SU3FASTER)
sum+=a.e00*b.e00;
sum+=a.e01*b.e10;
sum+=a.e02*b.e20;
sum+=a.e10*b.e01;
sum+=a.e11*b.e11;
sum+=a.e12*b.e21;
sum+=a.e20*b.e02;
sum+=a.e21*b.e12;
sum+=a.e22*b.e22;
#else
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++){
	sum += a.e[i][j] * b.e[j][i];
    }
#endif
  return sum;
}

/*--------------------------------------------------------------------*/

template<class Real>
__host__ __device__ inline void get_Q_from_VUadj(msun &Q, msun V, msun U){
  msun Omega = V * U.dagger();
  msun tmp = (Omega.dagger() - Omega)*complex(0.0, 0.5);
  tmp = tmp.subtraceunit();
  Q=tmp;
}

/*--------------------------------------------------------------------*/
/* Get the coefficients f of the expansion of exp(iQ)                
   and if do_bs = true
   get the coefficients b1 and b2 needed for the expansion of d exp(iQ)/dt */

template<class Real>
__host__ __device__ inline void get_fs_and_bs_from_Qs( complex f[3], complex b1[3], complex b2[3], msun &Q, msun &QQ, int &do_bs ){
  msun QQQ;
  Real trQQQ, trQQ, c0, c1;
  Real c0abs, c0max, theta;
  Real eps, sqtwo = sqrt(2.);
  Real u, w, u_sq, w_sq, xi0, xi1;
  Real cosu, sinu, cosw, sin2u, cos2u, ucosu, usinu, ucos2u, usin2u;//, sinw;
  Real denom;

  Real r_1_re[3], r_1_im[3], r_2_re[3], r_2_im[3];
  Real b_denom;

  QQQ = Q * QQ;
  trQQ  = QQ.realtrace();
  trQQQ = QQQ.realtrace();
  c0 = trQQQ / 3.0;
  c1 = trQQ * 0.5;
  
  if( c1 < 4.0e-3  ) 
    { // RGE: set to 4.0e-3 (CM uses this value). I ran into nans with 1.0e-4
      // =======================================================================
      // 
      // Corner Case 1: if c1 < 1.0e-4 this implies c0max ~ 3x10^-7
      //    and in this case the division c0/c0max in arccos c0/c0max can be undefined
      //    and produce NaN's
      
      // In this case what we can do is get the f-s a different way. We go back to basics:
      //
      // We solve (using maple) the matrix equations using the eigenvalues 
      //
      //  [ 1, q_1, q_1^2 ] [ f_0 ]       [ exp( iq_1 ) ]
      //  [ 1, q_2, q_2^2 ] [ f_1 ]   =   [ exp( iq_2 ) ]
      //  [ 1, q_3, q_3^2 ] [ f_2 ]       [ exp( iq_3 ) ]
      //
      // with q_1 = 2 u w, q_2 = -u + w, q_3 = - u - w
      // 
      // with u and w defined as  u = sqrt( c_1/ 3 ) cos (theta/3)
      //                     and  w = sqrt( c_1 ) sin (theta/3)
      //                          theta = arccos ( c0 / c0max )
      // leaving c0max as a symbol.
      //
      //  we then expand the resulting f_i as a series around c0 = 0 and c1 = 0
      //  and then substitute in c0max = 2 ( c_1/ 3)^(3/2)
      //  
      //  we then convert the results to polynomials and take the T and imaginary parts:
      //  we get at the end of the day (to low order)
      
      //                  1    2 
      //   f0[re] := 1 - --- c0  + h.o.real()
      //                 720     
      //
      //               1       1           1        2 
      //   f0[im] := - - c0 + --- c0 c1 - ---- c0 c1   + h.o.real()
      //               6      120         5040        
      //
      //
      //              1        1            1        2 
      //   f1[re] := -- c0 - --- c0 c1 + ----- c0 c1  +  h.o.real()
      //             24      360         13440        f
      //
      //                 1       1    2    1     3    1     2
      //   f1[im] := 1 - - c1 + --- c1  - ---- c1  - ---- c0   + h.o.real()
      //                 6      120       5040       5040
      //
      //               1   1        1    2     1     3     1     2
      //   f2[re] := - - + -- c1 - --- c1  + ----- c1  + ----- c0  + h.o.real()
      //               2   24      720       40320       40320    
      //
      //              1        1              1        2
      //   f2[im] := --- c0 - ---- c0 c1 + ------ c0 c1  + h.o.real()
      //             120      2520         120960
      
      //  We then express these using Horner's rule for more stable evaluation.
      // 
      //  to get the b-s we use the fact that
      //                                      b2_i = d f_i / d c0
      //                                 and  b1_i = d f_i / d c1
      //
      //  where the derivatives are partial derivatives
      //
      //  And we just differentiate the polynomials above (keeping the same level
      //  of truncation) and reexpress that as Horner's rule
      // 
      //  This clearly also handles the case of a unit gauge as no c1, u etc appears in the 
      //  denominator and the arccos is never taken. In this case, we have the results in 
      //  the raw c0, c1 form and we don't need to flip signs and take complex conjugates.
      //
      //  (not CD) I checked the expressions below by taking the difference between the Horner forms
      //  below from the expanded forms (and their derivatives) above and checking for the
      //  differences to be zero. At this point in time maple seems happy.
      //  ==================================================================
          
      f[0].real() = 1. - c0*c0 / 720.;
      f[0].imag() = -(c0/6.)*(1. - (c1/20.)*(1. - (c1/42.))) ;
      
      f[1].real() =  c0/24.*(1. - c1/15.*(1. - 3.*c1/112.)) ;
      f[1].imag() =  1.-c1/6.*(1. - c1/20.*(1. - c1/42.)) - c0*c0/5040. ;
      
      f[2].real() = 0.5*(-1. + c1/12.*(1. - c1/30.*(1. - c1/56.)) + c0*c0/20160.);
      f[2].imag() = 0.5*(c0/60.*(1. - c1/21.*(1. - c1/48.)));
      
      if( do_bs ) {
	//  partial f0/ partial c0
	b2[0].real() = -c0/360.;
	b2[0].imag() =  -(1./6.)*(1.-(c1/20.)*(1.-c1/42.));
        
	// partial f0 / partial c1
	//
	b1[0].real() = 0;
	b1[0].imag() = (c0/120.)*(1.-c1/21.);
        
	// partial f1 / partial c0
	//
	b2[1].real() = (1./24.)*(1.-c1/15.*(1.-3.*c1/112.));
	b2[1].imag() = -c0/2520.;
	
        
	// partial f1 / partial c1
	b1[1].real() = -c0/360.*(1. - 3.*c1/56. );
	b1[1].imag() = -1./6.*(1.-c1/10.*(1.-c1/28.));
        
	// partial f2/ partial c0
	b2[2].real() = 0.5*c0/10080.;
	b2[2].imag() = 0.5*(  1./60.*(1.-c1/21.*(1.-c1/48.)) );
        
	// partial f2/ partial c1
	b1[2].real() = 0.5*(  1./12.*(1.-(2.*c1/30.)*(1.-3.*c1/112.)) ); 
	b1[2].imag() = 0.5*( -c0/1260.*(1.-c1/24.) );
        
      } // do_bs
    }
  else 
    { 
      // =======================================================================
      // Normal case: Do as per Morningstar-Peardon paper
      // =======================================================================

      c0abs = abs( c0 );
      c0max = 2.0 * pow( c1/3., 1.5);
      
      // =======================================================================
      // Now work out theta. In the paper the case where c0 -> c0max even when c1 is reasonable 
      // Has never been considered, even though it can arise and can cause the arccos function
      // to fail
      // Here we handle it with series expansion
      // =======================================================================
      eps = (c0max - c0abs)/c0max;
      
      if( eps < 0 ) {
	// =====================================================================
	// Corner Case 2: Handle case when c0abs is bigger than c0max. 
	// This can happen only when there is a rounding error in the ratio, and that the 
	// ratio is really 1. This implies theta = 0 which we'll just set.
	// =====================================================================
	theta = 0;
      }
      else if ( eps < 1.0e-3 ) {
	// =====================================================================
	// Corner Case 3: c0->c0max even though c1 may be actually quite reasonable.
	// The ratio |c0|/c0max -> 1 but is still less than one, so that a 
	// series expansion is possible.
	// SERIES of acos(1-epsilon): Good to O(eps^6) or with this cutoff to O(10^{-18}) Computed with Maple.
	//  BTW: 1-epsilon = 1 - (c0max-c0abs)/c0max = 1-(1 - c0abs/c0max) = +c0abs/c0max
	//
	// ======================================================================
	theta = sqtwo*sqrt(eps)*( 1 + ( (1./12.) + ( (3./160.) + ( (5./896.) + ( (35./18432.) + (63./90112.)*eps ) *eps) *eps) *eps) *eps);
      } 
      else {  
	// 
	theta = acos( c0abs/c0max );
      }
          
      u = sqrt(c1/3.)*cos(theta/3.);
      w = sqrt(c1)*sin(theta/3.);
      
      u_sq = u*u;
      w_sq = w*w;

      if( fabs(w) < 0.05 ) { 
	xi0 = 1. - (1./6.)*w_sq*( 1. - (1./20.)*w_sq*( 1. - (1./42.)*w_sq ) );
      }
      else {
	xi0 = sin(w)/w;
      }
      
      if( do_bs) {
	
	if( fabs(w) < 0.05 ) { 
	  xi1 = -( 1./3. - (1./30.)*w_sq*( 1. - (1./28.)*w_sq*( 1. - (1./54.)*w_sq ) ) );
	}
	else { 
	  xi1 = cos(w)/w_sq - sin(w)/(w_sq*w);
	}
      }

      cosu = cos(u);
      sinu = sin(u);
      cosw = cos(w);
      //sinw = sin(w);
      sin2u = sin(2*u);
      cos2u = cos(2*u);
      ucosu = u*cosu;
      usinu = u*sinu;
      ucos2u = u*cos2u;
      usin2u = u*sin2u;
      
      denom = 9.*u_sq - w_sq;

      {
	Real subexp1, subexp2, subexp3;

	subexp1 = u_sq - w_sq;
	subexp2 = 8*u_sq*cosw;
	subexp3 = (3*u_sq + w_sq)*xi0;
	
	f[0].real() = ( (subexp1)*cos2u + cosu*subexp2 + 2*usinu*subexp3 ) / denom ;
	f[0].imag() = ( (subexp1)*sin2u - sinu*subexp2 + 2*ucosu*subexp3 ) / denom ;

      }
      {
	Real subexp;
	
	subexp = (3*u_sq -w_sq)*xi0;
	
	f[1].real() = (2*(ucos2u - ucosu*cosw)+subexp*sinu)/denom;
	f[1].imag() = (2*(usin2u + usinu*cosw)+subexp*cosu)/denom;
      }
      {
	Real subexp;

	subexp=3*xi0;
      
	f[2].real() = (cos2u - cosu*cosw -usinu*subexp) /denom ;
	f[2].imag() = (sin2u + sinu*cosw -ucosu*subexp) /denom ;
      }

      if( do_bs )
	{
	  {
	      Real subexp1, subexp2, subexp3;
	      //          r_1[0]=Double(2)*cmplx(u, u_sq-w_sq)*exp2iu
	      //          + 2.0*expmiu*( cmplx(8.0*u*cosw, -4.0*u_sq*cosw)
	      //              + cmplx(u*(3.0*u_sq+w_sq),9.0*u_sq+w_sq)*xi0 );
	      
	      subexp1 = u_sq - w_sq;
	      subexp2 = 8.*cosw + (3.*u_sq + w_sq)*xi0 ;
	      subexp3 = 4.*u_sq*cosw - (9.*u_sq + w_sq)*xi0 ;
	      
	      r_1_re[0] = 2.*(ucos2u - sin2u *(subexp1)+ucosu*( subexp2 )- sinu*( subexp3 ) );
	      r_1_im[0] = 2.*(usin2u + cos2u *(subexp1)-usinu*( subexp2 )- cosu*( subexp3 ) );
	      
	  }
	  {
	      Real subexp1, subexp2;

	      // r_1[1]=cmplx(2.0, 4.0*u)*exp2iu + expmiu*cmplx(-2.0*cosw-(w_sq-3.0*u_sq)*xi0,2.0*u*cosw+6.0*u*xi0);
	      
	      subexp1 = cosw+3.*xi0;
	      subexp2 = 2.*cosw + xi0*(w_sq - 3.*u_sq);
	      
	      r_1_re[1] = 2.*((cos2u - 2.*usin2u) + usinu*subexp1) - cosu*subexp2;
	      r_1_im[1] = 2.*((sin2u + 2.*ucos2u) + ucosu*subexp1) + sinu*subexp2;
          }
	  {
	    Real subexp;
	    // r_1[2]=2.0*timesI(exp2iu)  +expmiu*cmplx(-3.0*u*xi0, cosw-3*xi0);
	    
	    subexp = cosw - 3.*xi0;
	    r_1_re[2] = -2.*sin2u -3.*ucosu*xi0 + sinu*subexp;
	    r_1_im[2] = 2.*cos2u  +3.*usinu*xi0 + cosu*subexp;
	  }
          
	  {
	    Real subexp;
	    //r_2[0]=-2.0*exp2iu + 2*cmplx(0,u)*expmiu*cmplx(cosw+xi0+3*u_sq*xi1,
	    //                                                 4*u*xi0);
	    
	    subexp = cosw + xi0 + 3.*u_sq*xi1;
	    r_2_re[0] = -2.*(cos2u + u*( 4.*ucosu*xi0 - sinu*subexp) );
	    r_2_im[0] = -2.*(sin2u - u*( 4.*usinu*xi0 + cosu*subexp) );
	  }
	  {
	    Real subexp;
          
	    // r_2[1]= expmiu*cmplx(cosw+xi0-3.0*u_sq*xi1, 2.0*u*xi0);
	    // r_2[1] = timesMinusI(r_2[1]);
	    
	    subexp =  cosw + xi0 - 3.*u_sq*xi1;
	    r_2_re[1] =  2.*ucosu*xi0 - sinu*subexp;
	    r_2_im[1] = -2.*usinu*xi0 - cosu*subexp;
	    
	  }
	  {
	    Real subexp;
	    //r_2[2]=expmiu*cmplx(xi0, -3.0*u*xi1);
	    
	    subexp = 3.*xi1;
            
	    r_2_re[2] =    cosu*xi0 - usinu*subexp ;
	    r_2_im[2] = -( sinu*xi0 + ucosu*subexp ) ;
	  }
          
	  b_denom=2.*denom*denom;
          
	  {
	    Real subexp1, subexp2, subexp3;
	    int j;

	    subexp1 = 2.*u;
	    subexp2 = 3.*u_sq - w_sq;
	    subexp3 = 2.*(15.*u_sq + w_sq);
	    
	    for(j=0; j < 3; j++) { 
	      
	      b1[j].real()=( subexp1*r_1_re[j] + subexp2*r_2_re[j] - subexp3*f[j].real() )/b_denom;
	      b1[j].imag()=( subexp1*r_1_im[j] + subexp2*r_2_im[j] - subexp3*f[j].imag() )/b_denom;
	    }
	  }
	  {
	    Real subexp1, subexp2;
	    int j;
	    
	    subexp1 = 3.*u;
	    subexp2 = 24.*u;
	    
	    for(j=0; j < 3; j++) { 
	      b2[j].real()=( r_1_re[j] - subexp1*r_2_re[j] - subexp2 * f[j].real() )/b_denom;
	      b2[j].imag()=( r_1_im[j] - subexp1*r_2_im[j] - subexp2 * f[j].imag() )/b_denom;
	    }
	  }
	  
	  // Now flip the coefficients of the b-s
	  if( c0 < 0 ) 
	    {
	      //b1_site[0] = conj(b1_site[0]);
	      b1[0].imag() *= -1;
	      
	      //b1_site[1] = -conj(b1_site[1]);
	      b1[1].real() *= -1;
	      
	      //b1_site[2] = conj(b1_site[2]);
	      b1[2].imag() *= -1;
	      
	      //b2_site[0] = -conj(b2_site[0]);
	      b2[0].real() *= -1;
	      
	      //b2_site[1] = conj(b2_site[1]);
	      b2[1].imag() *= -1;
	      
	      //b2_site[2] = -conj(b2_site[2]);
	      b2[2].real() *= -1;
	    }
	} // end of if (do_bs)
      
      // Now when everything is done flip signs of the b-s (can't do this before
      // as the unflipped f-s are needed to find the b-s
      
      if( c0 < 0 ) {
	
	// f[0] = conj(f[0]);
	f[0].imag() *= -1;
        
	//f[1] = -conj(f[1]);
	f[1].real() *= -1;
        
	//f[2] = conj(f[2]);
	f[2].imag() *= -1;
        
      }
    } // End of if( corner_caseP ) else {}
}

template<class Real>
__host__ __device__ inline void quadr_comb( msun &TT, msun Q, msun QQ, complex f[3]){

  /*   T = f[0] + f[1]*Q + f[2]*QQ */
  TT = msun::zero();
#if (NCOLORS == 3) && defined(SU3FASTER)
  TT.e00 = f[0];
  TT.e11 = f[0];
  TT.e22 = f[0];
#elif (NCOLORS == 3)
  TT.e[0][0] = f[0];
  TT.e[1][1] = f[0];
  TT.e[2][2] = f[0];
#else
  for(int i=0;i<3;i++) TT.e[i][i] = f[0];
#endif
 TT += Q * f[1];
 TT += QQ * f[2];
}

template<class Real>
__host__ __device__ inline void exp_iQ( msun &TT, msun &Q ){
  complex f[3];
  complex b1[3], b2[3];
  f[0]=complex::zero();
  f[1]=complex::zero();
  f[2]=complex::zero();
  int do_bs = 0;
  msun QQ = Q * Q;
  get_fs_and_bs_from_Qs( f, b1, b2, Q, QQ, do_bs);
  quadr_comb<Real>( TT, Q, QQ, f);
}

/*--------------------------------------------------------------------*/

/* Do Morningstar-Peardon stout smearing to construct unitary W from
   the smeared link V and the unsmeared link U.

   Smearing applies to any scheme and not just the APE smearing
   described in MP.  For the general smearing case we still have 
 
      V = k*U + C 

   where C is the analog of the sum of APE terms in MP, U is the
   unsmeared link and k is any constant.  Note that Q in the first
   stout smearing step is the traceless hermitian part of -i V Uadj,
   which is the same as the traceless hermitian part of -i C Uadj, as
   used in MP. */

template<class Real>
__host__ __device__ inline void stout_smear(msun &W, msun V, msun U){

  msun Q, tmp;
  get_Q_from_VUadj<Real>(Q, V, U);
  /* tmp = exp(iQ) */
  exp_iQ<Real>( tmp, Q );
  /* W = exp(iQ) U */
  W = tmp * U;


/*
if(threadIdx.x+blockDim.x*blockIdx.x ==0){
  tmp.print();
  (tmp.det()).print();
  eesu3<Real>(Q);
  Q.print();
  (Q.det()).print();
}*/
}







template<class Real>
struct StoutArg{
  complex *arrayin;
  complex *arrayout;
  Real w;
};



template <bool UseTex, ArrayType atype, class Real>
__global__ void kernel_Stout(StoutArg<Real> arg, int mu){  
        
  int id = INDEX1D();
  if(id >= DEVPARAMS::Volume) return;
  
  int x[NDIMS];
  Index_ND_NM(id, x);
  int mustride = DEVPARAMS::Volume;
  int offset = DEVPARAMS::size;
  
  int muvolume = mu * mustride;
  msun link;
  msun staple = msu3::zero();
  for(int nu = 0; nu < NDIMS-1; nu++)  if(mu != nu) {
    int dx[NDIMS] = {0}; 
    int nuvolume = nu * mustride;
    link = GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin,  id + nuvolume, offset);
    dx[nu]++;
    link *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_ND_Neig_NM(x, dx) + muvolume, offset); 
    dx[nu]--;
    dx[mu]++;
    link *= GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.arrayin, Index_ND_Neig_NM(x, dx) + nuvolume, offset);
    staple += link;

    dx[mu]--;
    dx[nu]--;
    link = GAUGE_LOAD_DAGGER<UseTex, atype, Real>( arg.arrayin,  Index_ND_Neig_NM(x, dx) + nuvolume, offset);  
    link *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_ND_Neig_NM(x, dx)  + muvolume, offset);
    dx[mu]++;
    link *= GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin, Index_ND_Neig_NM(x, dx) + nuvolume, offset);
    staple += link;
  }
  msun U = GAUGE_LOAD<UseTex, atype, Real>( arg.arrayin,  id + muvolume, offset);
  msun sm;
  stout_smear<Real>( sm, staple*arg.w, U );
  GAUGE_SAVE<atype, Real>( arg.arrayout, sm, id + muvolume, offset);
}
  




  





template <bool UseTex, ArrayType atypein, class Real> 
class ApplyStout: Tunable{
private:
   gauge arrayin;
   gauge arrayout;
   StoutArg<Real> arg;
   int size;
   double timesec;
   int mu;
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
      kernel_Stout<UseTex, atypein, Real><<<tp.grid,tp.block, 0, stream>>>(arg, mu);
  }
public:
   ApplyStout(gauge &arrayin, gauge & arrayout, Real w):arrayin(arrayin),arrayout(arrayout){
    size = 1;
    //Number of threads is equal to the number of space points!
    for(int i=0;i<NDIMS;i++){
      size *= PARAMS::Grid[i];
    } 
    size = size;
    timesec = 0.0;
    arg.arrayin = arrayin.GetPtr();
    arg.arrayout = arrayout.GetPtr();
    arg.w = w;
    mu = 0;
  }
  ~ApplyStout(){};
  
  void SetDir(int muin){ mu = muin;};

  double time(){return timesec;}
  void stat(){ COUT << "Apply Stout smearing:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  long long flop() const { return 0;}
  long long bytes() const { return 0;}
  double flops(){ return ((double)flop() * 1.0e-9) / timesec;}
  double bandwidth(){ return (double)bytes() / (timesec * (double)(1 << 30));}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    for(int i=0; i<NDIMS-1; i++) vol << PARAMS::Grid[i] << "x";
    vol << PARAMS::Grid[NDIMS-1];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string typear = arrayin.ToStringArrayType()+arrayout.ToStringArrayType();
    return TuneKey(vol.str().c_str(), typeid(*this).name(), typear.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() {}
  
  void Run(const cudaStream_t &stream){
  #ifdef TIMMINGS
      mtime.start();
  #endif
      apply(stream);    
  #ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC( );
      mtime.stop();
      timesec = mtime.getElapsedTimeInSec();
  #endif
  }
  void Run(){return Run(0);}
};




template<class Real>
void ApplyStoutinSpace(gauge array, Real w, int steps){
  cout << "Apply Stout Smearing in Space with w = " << w << " steps = " << steps << endl;
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  gauge arrayout(array.Type(), Device, NDIMS*PARAMS::Volume, array.EvenOdd());
  arrayout.Copy(array);
  const ArrayType atypein = SOA;
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyStout<true, atypein, Real> Stout(array, arrayout, w);
    for(int st = 0; st < steps; st++){
      for(int mu = 0; mu < NDIMS-1; mu++){
        Stout.SetDir(mu);
        Stout.Run();
      }
      array.Copy(arrayout);
    }
    Stout.stat();
  }
  else{
    ApplyStout<false, atypein, Real> Stout(array, arrayout, w);
    for(int st = 0; st < steps; st++){
      for(int mu = 0; mu < NDIMS-1; mu++){
        Stout.SetDir(mu);
        Stout.Run();
      }
      array.Copy(arrayout);
    }
    Stout.stat();
  }
  arrayout.Release();
}
template void ApplyStoutinSpace<float>(gauges array, float w, int steps);
template void ApplyStoutinSpace<double>(gauged array, double w, int steps);









template<class Real>
void ApplyStoutinTime(gauge array, Real w, int steps){
  cout << "Apply Stout Smearing in Time with w = " << w << " steps = " << steps << endl;
  if(array.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if(array.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");
    
  gauge arrayout(array.Type(), Device, 4*PARAMS::Volume, array.EvenOdd());
  arrayout.Copy(array);
  const ArrayType atypein = SOA;
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(array.GetPtr(), true);
    ApplyStout<true, atypein, Real> Stout(array, arrayout, w);
    Stout.SetDir(NDIMS-1);
    for(int st = 0; st < steps; st++){
      Stout.Run();
      array.Copy(arrayout);
    }
    Stout.stat();
  }
  else{
    ApplyStout<false, atypein, Real> Stout(array, arrayout, w);
    Stout.SetDir(NDIMS-1);
    for(int st = 0; st < steps; st++){
      Stout.Run();
      array.Copy(arrayout);
    }
    Stout.stat();
  }
  arrayout.Release();
}
template void ApplyStoutinTime<float>(gauges array, float w, int steps);
template void ApplyStoutinTime<double>(gauged array, double w, int steps);






}
