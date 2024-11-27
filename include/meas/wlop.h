
#ifndef WLOP_H
#define WLOP_H

namespace CULQCD {

template <class Real>
void CalcWilsonLoop(gauge array, gauge fieldOp, complex *res, int radius,
                    int Tmax, int mu, int opN);

template <class Real>
void CalcWLOPs(gauge array, gauge fieldOp, int radius, int mu, int opN);

template <class Real>
void PlaquetteSTField(gauge array, complex *plaq, complex *meanplaq);
template <class Real>
void PlaquetteSTField(gauge array, Real *plaq, Real *meanplaq);

template <class Real>
void CalcWLOPs_dg(gauge array, gauge fieldOp, int radius, int mu, int opN);

template <class Real>
void CalcFieldWilsonLoop_AllOP(gauge array, gauge fieldOp, Real *plaqfield,
                               Real *wloop, Real *field, int radius, int Tmax,
                               int mu, int opN, int nx, int ny, bool planexy);

template <class Real>
void CalcFieldWilsonLoop_dg(gauge array, gauge fieldOp, Real *plaqfield,
                            Real *wloop, Real *field, int radius, int Tmax,
                            int mu, int opN, int nx, int ny, bool planexy);

template <class Real>
void CalcWilsonLoop(gauge array, gauge fieldOp, complex *res, int radius,
                    int Tmax, int mu, int opN);

template <class Real>
void CalcWLOPs33(gauge array, gauge fieldOp, int radius, int mu, int opN);

template <class Real>
void CalcWLOPsD1(gauge array, gauge fieldOp, int radius, int mu, int opN);

template <class Real>
void CalcWLOPsLDL(gauge array, gauge fieldOp, int radius, int mu, int opN);

template <class Real>
void CalcWLOPsST2(gauge array, gauge fieldOp, int radius, int mu, int opN);

template <class Real>
void CalcWLOPs_dg_33(gauge array, gauge fieldOp, int radius, int mu, int opN);

extern __constant__ int DEV_Ops[2];
extern __constant__ int DEV_OpComps[8];
extern __constant__ int DEV_OpPos[8];
} // namespace CULQCD

#endif
