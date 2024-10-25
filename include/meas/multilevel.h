
#ifndef MULTILEVEL_H
#define MULTILEVEL_H

#include <constants.h>

namespace CULQCD {

template <class Real> struct Multi_Level_Params {
  gauge array;
  RNG randstates;
  complex *trace;
  bool SpRes;
  complex *res;
  int nhb;
  int novr;
  int mhb;
  int movr;
  int nhit;
  int lvl0, lvl1;
  int nit;
  int mit;
  int rmin, rmax;
  int nl0;
  int nl1;

  void Print() {
    COUT << "MultiLevel Params:" << endl;
    COUT << "\t nhb: " << nhb << endl;
    COUT << "\t mhb: " << mhb << endl;
    COUT << "\t novr: " << novr << endl;
    COUT << "\t movr: " << movr << endl;
    COUT << "\t nhit: " << nhit << endl;
    COUT << endl;

    COUT << "\t lvl0: " << lvl0 << endl;
    COUT << "\t lvl1: " << lvl1 << endl;
    COUT << "\t nit: " << nit << endl;
    COUT << "\t mit: " << mit << endl;
    COUT << "\t nl0: " << nl0 << endl;
    COUT << "\t nl1: " << nl1 << endl;
    COUT << endl;
    COUT << "\t Rmin: " << rmin << endl;
    COUT << "\t Rmax: " << rmax << endl;
  }
};

template <class Real> void ApplyMultiLevel(Multi_Level_Params<Real> &mlp);

template <class Real>
void CalcChromoField_ML(complex *ploop, complex *plaqfield, Real *field,
                        int radius, int nx, int ny, bool chargeplane);

template <class Real>
void PlaquetteField_3D(gauge array, complex *plaq, complex *meanplaq);
}

#endif
