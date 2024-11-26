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
#include <vector>

using namespace std;
using namespace CULQCD;

template <class Real, ArrayType mygaugein>
void runHeatBath(int argc, char **argv);

// NEEDS: export CULQCD_RESOURCE_PATH="path to folder where the tuning
// parameters are saved..."
int main(int argc, char **argv) {
  cout << sizeof(curandStateMRG32k3a) << ":::::" << sizeof(curandStateXORWOW)
       << endl;
  cout << sizeof(float) << ":::::" << sizeof(double) << endl;

  COUT << "####################################################################"
          "###########" << endl;
  COUT << "Start generating SU(" << NCOLORS << ") gauge configurations in " << NDIMS << " spacetime dimensions" << endl;
  const ArrayType mygauge = SOA; // SOA/SOA12/SOA8 for SU(3) and SOA for N>3
  runHeatBath<double, mygauge>(argc, argv);
  COUT << "####################################################################"
          "###########" << endl;
  EndCULQCD(0);
  COUT << "####################################################################"
          "###########" << endl;
  exit(0);
}

// heatbath test module
template <class Real, ArrayType mygaugein>
void runHeatBath(int argc, char **argv) {
  if((argc-1) != (NDIMS+2)) {
        errorCULQCD("Number of input arguments is %d, should be (NDIMS + 2) = %d", argc - 1, NDIMS + 2);
  }
  vector<int> lattice_size;
  lattice_size.reserve(NDIMS);
  for(int i=1; i<=NDIMS; i++) lattice_size.push_back(atoi(argv[i]));
  //int ns = atoi(argv[1]);
  //int nt = atoi(argv[2]);
  float beta0 = atof(argv[NDIMS+1]);
  PARAMS::UseTex = false;
  int ntraj = atoi(argv[NDIMS+2]);


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
  //SETPARAMS(PARAMS::UseTex, beta0, ns, ns, ns, nt, true);
  SETPARAMS(PARAMS::UseTex, beta0, lattice_size, true);

  gauge conf(mygaugein, Device, PARAMS::Volume * NDIMS, true);
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
