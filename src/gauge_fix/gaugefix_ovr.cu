
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <typeinfo>

#include <comm_mpi.h>
#include <complex.h>
#include <cudaAtomic.h>
#include <cuda_common.h>
#include <gaugefix/gaugefix.h>
#include <matrixsun.h>

#include <constants.h>
#include <device_load_save.h>
#include <exchange.h>
#include <index.h>
#include <launch_kernel.cuh>
#include <reduction.h>
#include <reunitarize.h>
#include <sharedmemtypes.h>
#include <texture.h>
#include <texture_host.h>
#include <timer.h>
#include <tune.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace CULQCD {

#ifdef USE_GAUGE_FIX

#include "gaugefix_ovr.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, int DIR, ArrayType atype, class Real>
complex PerformGaugeFixOVR(gauge _pgauge, Real relax_boost, Real stopvalue,
                           int maxsteps, int reunit_interval, int verbose) {
  COUT << "####################################################################"
          "###########"
       << endl;
  string my_name = "";
  if (DIR == 4)
    my_name = "Landau gauge fixing";
  else if (DIR == 3)
    my_name = "Coulomb gauge fixing";
  else {
    COUT << "DIR can only be 3, for Coulomb, or 4, for Landau.\nNo gauge "
            "fixing applied."
         << endl;
    return complex::make_complex(9999., 999999999999.);
  }
  if (_pgauge.EvenOdd() == false) {
    COUT << "Only implemented for EvenOdd arrays.\nNo gauge fixing applied."
         << endl;
    return complex::make_complex(9999., 999999999999.);
  }
  if (reunit_interval < 1)
    reunit_interval = 1;
  if (verbose < 1)
    verbose = maxsteps;
  COUT << "Applying " << my_name << "." << endl;
  COUT << "\tOverrelaxation boost parameter: " << relax_boost << endl;
  COUT << "\tStop criterium: " << stopvalue << endl;
  COUT << "\tMaximum number of iterations: " << maxsteps << endl;
  COUT << "\tReunitarize at every " << reunit_interval << " steps" << endl;
  COUT << "\tPrint convergence results at every " << verbose << " steps"
       << endl;

  Timer gfltime;
  gfltime.start();
  //------------------------------------------------------------------------
  // Bind array to texture if PARAMS::UseTex is true
  // The bool here means bind(true) or unbind(false)
  //------------------------------------------------------------------------
  GAUGE_TEXTURE(_pgauge.GetPtr(), true);
#ifdef MULTI_GPU
  cudaStream_t stream;
  if (numnodes() > 1) {
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
    // Update ghost links just in case...
    for (int par = 0; par < 2; par++)
      for (int mu = 0; mu < 4; mu++)
        Exchange_gauge_border_links_gauge(_pgauge, mu, par);
  }
#endif

#ifdef USE_CUDA_CUB
  GaugeFixQualityCUB<DIR, UseTex, atype, Real> quality(_pgauge);
#else
  GaugeFixQuality<DIR, UseTex, atype, Real> quality(_pgauge);
#endif
  Reunitarize<Real> reu(_pgauge);
#ifdef MULTI_GPU
  GaugeFix_Interior<DIR, UseTex, atype, Real> fix(_pgauge, relax_boost);
  GaugeFix_Border<DIR, UseTex, atype, Real> fixborder(_pgauge, relax_boost);
#else
  GaugeFix_SingleNode<DIR, UseTex, atype, Real> fixsingle(_pgauge, relax_boost);
#endif
  //------------------------------------------------------------------------
  // Measure initial gauge fixing quality
  //------------------------------------------------------------------------
  COUT << "................................................." << endl;
  complex data;
  Real diff, oldvalue;
  data = quality.Run();
  diff = abs(data.real());
  printfCULQCD("Iter: %d\tFg = %.12e\ttheta = %.12e\tDelta = %.12e\n", 0,
               data.real(), data.imag(), diff);
  oldvalue = data.real();
  long long flopp = quality.flop();
  long long bytes = quality.bytes();
  //------------------------------------------------------------------------
  // Do gauge fix
  //------------------------------------------------------------------------
  int iterations = 0;
  for (int iter = 1; iter <= maxsteps; iter++) {
    iterations++;
#ifndef MULTI_GPU
    fixsingle.Run(0);
    flopp += 2 * fixsingle.flop();
    bytes += 2 * fixsingle.bytes();
#else
    if (numnodes() == 1) {
      /*fixsingle.Run(0);
      flopp += 2*fixsingle.flop();
      bytes += 2*fixsingle.bytes();*/
      for (int oddbit = 0; oddbit < 2; oddbit++) {
        fix.Run(0, oddbit);
        flopp += fix.flop();
        bytes += fix.bytes();
      }
    } else {
      for (int oddbit = 0; oddbit < 2; oddbit++) {
        fixborder.Run(0, oddbit);
        cudaDeviceSynchronize();
        fix.Run(stream, oddbit);
        flopp += fixborder.flop();
        bytes += fixborder.bytes();
        flopp += fix.flop();
        bytes += fix.bytes();
      }
    }
#endif
    if ((iter % reunit_interval) == 0) {
      reu.Run();
      flopp += reu.flop();
      bytes += reu.bytes();
    }
    data = quality.Run();
    diff = abs(data.real() - oldvalue);
    if ((iter % verbose) == 0)
      printfCULQCD("Iter: %d\tFg = %.12e\ttheta = %.12e\tDelta = %.12e\n", iter,
                   data.real(), data.imag(), diff);
    oldvalue = data.real();
#ifdef USE_THETA_STOP_GAUGEFIX
    if (data.imag() < stopvalue)
      break;
#else
    if (diff < stopvalue)
      break;
#endif
    flopp += quality.flop();
    bytes += quality.bytes();
  }
  //------------------------------------------------------------------------
  // Reunitarize at the end if not yet reunitarized in last iteration
  //------------------------------------------------------------------------
  if ((iterations % reunit_interval) != 0) {
    reu.Run();
    flopp += reu.flop();
    bytes += reu.bytes();
  }
  if ((iterations % verbose) != 0)
    printfCULQCD("Iter: %d\tFg = %.12e\ttheta = %.12e\tDelta = %.12e\n",
                 iterations, data.real(), data.imag(), diff);
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    CUDA_SAFE_CALL(cudaStreamDestroy(stream));
    // Update ghost links
    for (int par = 0; par < 2; par++)
      for (int mu = 0; mu < 4; mu++)
        Exchange_gauge_border_links_gauge(_pgauge, mu, par);
  }
#endif
  //------------------------------------------------------------------------
  // Unbind texture
  //------------------------------------------------------------------------
  GAUGE_TEXTURE(_pgauge.GetPtr(), false);
  CUDA_SAFE_DEVICE_SYNC();
  COUT << "Finishing " << my_name << "..." << endl;
  gfltime.stop();
  double flops = ((double)flopp * 1.0e-9) / gfltime.getElapsedTimeInSec();
  ;
  double bdw =
      (double)bytes / (gfltime.getElapsedTimeInSec() * (double)(1 << 30));
  COUT << my_name << ": " << gfltime.getElapsedTimeInSec() << " s\t" << bdw
       << " GB/s\t" << flops << " GFlops" << endl;
  COUT << "####################################################################"
          "###########"
       << endl;
  return complex::make_complex(data.imag(), (Real)iterations);
}
template <int DIR, ArrayType atype, class Real>
complex PerformGaugeFixOVR(gauge _pgauge, Real relax_boost, Real stopvalue,
                           int maxsteps, int reunit_interval, int verbose) {
  if (PARAMS::UseTex)
    return PerformGaugeFixOVR<true, DIR, atype, Real>(
        _pgauge, relax_boost, stopvalue, maxsteps, reunit_interval, verbose);
  else
    return PerformGaugeFixOVR<false, DIR, atype, Real>(
        _pgauge, relax_boost, stopvalue, maxsteps, reunit_interval, verbose);
}
template <ArrayType atype, class Real>
complex PerformGaugeFixOVR(gauge _pgauge, int DIR, Real relax_boost,
                           Real stopvalue, int maxsteps, int reunit_interval,
                           int verbose) {
  if (DIR == 3)
    return PerformGaugeFixOVR<3, atype, Real>(
        _pgauge, relax_boost, stopvalue, maxsteps, reunit_interval, verbose);
  else
    return PerformGaugeFixOVR<4, atype, Real>(
        _pgauge, relax_boost, stopvalue, maxsteps, reunit_interval, verbose);
}
template <class Real>
complex PerformGaugeFixOVR(gauge _pgauge, int DIR, Real relax_boost,
                           Real stopvalue, int maxsteps, int reunit_interval,
                           int verbose) {
#if (NCOLORS == 3)
  if (_pgauge.Type() == SOA && _pgauge.EvenOdd())
    return PerformGaugeFixOVR<SOA, Real>(_pgauge, DIR, relax_boost, stopvalue,
                                         maxsteps, reunit_interval, verbose);
  else if (_pgauge.Type() == SOA12 && _pgauge.EvenOdd())
    return PerformGaugeFixOVR<SOA12, Real>(_pgauge, DIR, relax_boost, stopvalue,
                                           maxsteps, reunit_interval, verbose);
  else if (_pgauge.Type() == SOA8 && _pgauge.EvenOdd())
    return PerformGaugeFixOVR<SOA8, Real>(_pgauge, DIR, relax_boost, stopvalue,
                                          maxsteps, reunit_interval, verbose);
  else
    errorCULQCD("Only defined for array SOA/SOA12/SOA in even/odd format...\n");
  return complex::make_complex(9999., 999999999999.);
#else
  return PerformGaugeFixOVR<SOA, Real>(_pgauge, DIR, relax_boost, stopvalue,
                                       maxsteps, reunit_interval, verbose);
#endif
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
   @brief Apply Landau/Coulomb Gauge Fixing with Overrelaxation method
   @param array gauge field to be fixed
   @param DIR DIR=4 for Landau gauge Fixing and DIR=3 for Coulomb gauge fixing
   @param relax_boost Overrelaxation parameter
   @param stopvalue criterium to stop the method, precision
   @param maxsteps maximum number of iterations
   @param reunit_interval Reunitarize when iteration count is a multiple of this
   @param verbose print to screen Fg, theta and delta when iteration count is a
   multiple of this
*/
template <class Real>
complex GaugeFixingOvr(gauge _pgauge, int DIR, Real relax_boost, Real stopvalue,
                       int maxsteps, int reunit_interval, int verbose) {
  if (_pgauge.EvenOdd() == false) {
    COUT << "Only implemented for EvenOdd arrays.\nNo gauge fixing applied."
         << endl;
    return complex::make_complex(9999., 999999999999.);
  }
  return PerformGaugeFixOVR(_pgauge, DIR, relax_boost, stopvalue, maxsteps,
                            reunit_interval, verbose);
}
template complexs GaugeFixingOvr<float>(gauges _pgauge, int DIR,
                                        float relax_boost, float stopvalue,
                                        int maxsteps, int reunit_interval,
                                        int verbose);
template complexd GaugeFixingOvr<double>(gauged _pgauge, int DIR,
                                         double relax_boost, double stopvalue,
                                         int maxsteps, int reunit_interval,
                                         int verbose);

#endif
} // namespace CULQCD
