
#ifndef SMEAR_H
#define SMEAR_H

#include <constants.h>

namespace CULQCD {

////////////////////////////   APE SMEARING //////////////////////////////
template <class Real>
void ApplyAPEinSpace(gauge array, Real w, int steps, int nhits, Real tol);

template <class Real>
void ApplyAPEinTime(gauge array, Real w, int steps, int nhits, Real tol);

////////////////////////////   HYP SMEARING //////////////////////////////
class ParamHYP {
private:
  float alpha1;
  float alpha2;
  float alpha3;

public:
  ParamHYP();
  ParamHYP(float alpha1, float alpha2, float alpha3);
  ~ParamHYP();
  void setDefault();
  void set(float _alpha1, float _alpha2, float _alpha3);
  void print();
  void copyToGPUMem();
};

template <class Real>
void ApplyHYPinSpace(gauge array, int steps, int nhits, Real tol, ParamHYP hyp);

template <class Real>
void ApplyHYPinTime(gauge array, int steps, int nhits, Real tol, ParamHYP hyp);

////////////////////////////   STOUT SMEARING //////////////////////////////
template <class Real> void ApplyStoutinSpace(gauge array, Real w, int steps);

template <class Real> void ApplyStoutinTime(gauge array, Real w, int steps);

////////////////////////////   MULTIHIT SMEARING //////////////////////////////
template <class Real>
void ApplyMultiHit(gauge array, gauge arrayout, RNG &randstates, int nhit);

template <class Real>
void ApplyMultiHitSpace(gauge array, gauge arrayout, RNG &randstates, int nhit);

////////////////////////////   MULTIHIT EXTENDED SMEARING
/////////////////////////////////
template <class Real>
void ApplyMultiHitExt(gauge array, gauge arrayout, RNG &randstates, int nhit);
} // namespace CULQCD

#endif
