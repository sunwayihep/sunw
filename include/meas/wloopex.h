
#ifndef WL_FUNC_H
#define WL_FUNC_H

namespace CULQCD {

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/////      Calculate Sigma_g^+ states (4 for now)/////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
template <class Real> class Sigma_g_plus {
public:
  int Rmax;
  int Tmax;
  int opN;
  int totalOpN;
  Real *wloop;
  Real *wloop_h;
  size_t wloop_size;
  // Store all space line components
  gauge fieldOp;
  Sigma_g_plus(const int _Rmax, const int _Tmax) : Rmax(_Rmax), Tmax(_Tmax) {
    opN = 10;
    totalOpN = opN * opN;
    fieldOp.Set(SOA, Device, false);
    fieldOp.Allocate(PARAMS::Volume * opN);
    wloop_size = totalOpN * (Tmax + 1) * sizeof(Real);
    wloop = (Real *)dev_malloc(wloop_size);
    wloop_h = (Real *)safe_malloc(wloop_size);
  }
  ~Sigma_g_plus() {
    dev_free(wloop);
    host_free(wloop_h);
    fieldOp.Release();
  }
};
// Calculate the 4 types of space Wilson lines for a give radius and direction
// mu
template <class Real>
void CalcWLOPs_A0(gauge array, Sigma_g_plus<Real> *arg, int radius, int mu);

// Calculate the Wilson loop for 4 operators for a give radius and direction mu
template <class Real>
void CalcWilsonLoop_A0(gauge array, Sigma_g_plus<Real> *arg, int radius,
                       int mu);

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
///////////  OLD CODE...........................
////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

extern __constant__ int DEV_Ops[2];
extern __constant__ int DEV_OpComps[8];
extern __constant__ int DEV_OpPos[8];

template <class Real>
void CalcWilsonLoop_dg(gauge array, gauge fieldOp, Real *wloop, int radius,
                       int Tmax, int mu, int opN);

template <class Real>
void CalcWLOPs_dg_33(gauge array, gauge fieldOp, int radius, int mu, int opN);
} // namespace CULQCD

#endif
