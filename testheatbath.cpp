#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <culqcd.h>
//#include "gnuplot.h"

#include <stdio.h>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()
#include <iostream>

using namespace std;
using namespace CULQCD;

template <class Real, ArrayType mygaugein>
void heatBathTEST(int argc, char **argv);

// NEEDS: export CULQCD_RESOURCE_PATH="path to folder where the tuning
// parameters are saved..."
int main(int argc, char **argv) {
  cout << sizeof(curandStateMRG32k3a) << ":::::" << sizeof(curandStateXORWOW)
       << endl;
  cout << sizeof(float) << ":::::" << sizeof(double) << endl;

  COUT << "####################################################################"
          "###########" << endl;
  COUT << "Start generating SU(" << NCOLORS << ") gauge configurations" << endl;
  const ArrayType mygauge = SOA; // SOA/SOA12/SOA8 for SU(3) and SOA for N>3
  heatBathTEST<double, mygauge>(argc, argv);
  COUT << "####################################################################"
          "###########" << endl;
  EndCULQCD(0);
  COUT << "####################################################################"
          "###########" << endl;
  exit(0);
}

// heatbath test module
template <class Real, ArrayType mygaugein>
void heatBathTEST(int argc, char **argv) {
  int ns = atoi(argv[1]);
  int nt = atoi(argv[2]);
  float beta0 = atof(argv[3]);
  PARAMS::UseTex = false;
  int ntraj = atoi(argv[4]);

  // init the MPI environment, gpuid not used for multi-GPU, while can be set
  // for single-GPU run.
  // if TUNE_YES user must set export CULQCD_RESOURCE_PATH="path to folder where
  // the tuning parameters are saved..."
  initCULQCD(0, SUMMARIZE, TUNE_YES);

  // Create timer
  Timer t0;

  // Start timer
  t0.start();

  // Set Lattice Gauge Parameters and copy to Device constant memory
  // also sets some kernel launch parameters
  // true for verbosity
  //---------------------------------------------------------------------------------------
  SETPARAMS(PARAMS::UseTex, beta0, ns, ns, ns, nt, true);

  gauge conf(mygaugein, Device, PARAMS::Volume * 4, true);
  conf.Details();

  // init the random number generator
  RNG randstates;
  int rngSeed = 1234;
  randstates.Init(rngSeed);

  // Init() for Cold start and Init(randstates) for Hot start
  conf.Init();

  // prepare for heatbath
  HeatBath<Real> heatBath(conf, randstates);

  for (int i = 0; i < ntraj; ++i) {
    COUT << "==========Traj-" << i << "==========" << endl;
    Timer h0;
    h0.start();

    heatBath.Run();
    heatBath.stat();

    h0.stop();
    COUT << "Time for "
         << "traj "
         << "#" << i << ": " << h0.getElapsedTime() << " s" << endl;
    COUT << endl;

    PlaquetteCUB<Real> plaqCUB(conf);
    OnePolyakovLoop<Real> poly(conf);

    // Reunitarize gauge field
    Reunitarize<Real> reu(conf);
    reu.Run();

    // Calculate plaquette
    plaqCUB.Run();
    plaqCUB.printValue();
    COUT << endl;

    // Calculate polyakov loop
    poly.Run();
    poly.printValue();
    COUT << endl;

    // Print performance
    plaqCUB.stat();
    poly.stat();
    COUT << endl;
  }
  randstates.Release();

  conf.Release();
  t0.stop();
  COUT << "####################################################################"
          "###########" << endl;
  COUT << "Total Time: " << t0.getElapsedTime() << " s" << endl;
  COUT << "####################################################################"
          "###########" << endl;
  return;
}
