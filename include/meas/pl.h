
#ifndef PL_H
#define PL_H

namespace CULQCD {

template <class Real> void MatrixPloop(gauge array, gauge ploop);

template <class Real> complex Ploop(gauge array);

template <class Real>
void PotPL(gauge ploop, complex *pot, int radius, bool ppdagger);

template <class Real>
void PotPL3D(gauge ploop, complex *pot, int radius, int pts);
}

#endif
