
#ifndef CHROMOFIELD_H
#define CHROMOFIELD_H

namespace CULQCD {

template <class Real>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field, Real *pl,
                     int radius, int nx, int ny, bool ppdagger,
                     bool chargeplane);

template <class Real>
void Calc_WilsonSpaceLine(gauge array, gauge wlspl, int radius);

template <class Real>
void CalcFieldWilsonLoop_dg(gauge array, gauge wilson_spaceline,
                            Real *plaqfield, Real *wloop, Real *field,
                            int radius, int Tmax, int nx, int ny, bool planexy);

template <class Real>
void CalcChromoField(gauge ploop, complex *plaqfield, Real *field,
                     Real *reffield, int radius, int nx, int ny);

template <class Real>
void CalcChromoField(gauge ploop, complex *plaqfield, Real *field, int radius);

template <class Real>
void PlaquetteField(gauge array, complex *plaq, complex *meanplaq);

template <class Real> complex TracePloop(gauge array, complex *ploop);

template <class Real>
Real CalcRefChromoField(gauge ploop, complex *plaqfield, int radius);

template <class Real>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field,
                     int radius, int nx, int ny, bool ppdagger,
                     bool chargeplane);

template <class Real>
void CalcChromoField_QG(complex *ploop, complex *plaqfield, Real *field,
                        Real *value, int radius, int nx, int ny,
                        bool chargeplane);

template <class Real>
void CalcChromoField_GG(complex *ploop, complex *plaqfield, Real *field,
                        Real *value, int radius, int nx, int ny,
                        bool chargeplane);

template <class Real>
void CalcChromoField_G(complex *ploop, complex *plaqfield, Real *field,
                       Real *value, int nx, int ny);

template <class Real>
void CalcChromoField_Q(complex *ploop, complex *plaqfield, Real *field,
                       Real *value, int nx, int ny);
} // namespace CULQCD

#endif
