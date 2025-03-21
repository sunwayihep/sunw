#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <culqcd.h>
// #include "gnuplot.h"

#include <stdio.h>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()
#include <vector>

#include "io/lqcd_read_write.h"
#include "check_error/check_cuda.cuh"
#include "base/datatype/qcu_complex.cuh"
#include "precondition/even_odd_precondition.h"
#include "buffer_shape/data_shift.h"
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
       << endl;
  COUT << "Start generating SU(" << NCOLORS << ") gauge configurations in "
       << NDIMS << " spacetime dimensions" << endl;
  const ArrayType mygauge = SOA; // SOA/SOA12/SOA8 for SU(3) and SOA for N>3
  runHeatBath<double, mygauge>(argc, argv);
  COUT << "####################################################################"
       << endl;
  EndCULQCD(0);
  COUT << "####################################################################"
       << endl;
  exit(0);
}

// heatbath test module
template <class Real, ArrayType mygaugein>
void runHeatBath(int argc, char **argv) {
  if ((argc - 1) != (NDIMS + 2)) {
    errorCULQCD("Number of input arguments is %d, should be (NDIMS + 2) = %d\n",
                argc - 1, NDIMS + 2);
  }
  vector<int> lattice_size;
  lattice_size.reserve(NDIMS);
  for (int i = 1; i <= NDIMS; i++)
    lattice_size.push_back(atoi(argv[i]));
  float beta0 = atof(argv[NDIMS + 1]);
  PARAMS::UseTex = false;
  int ntraj = atoi(argv[NDIMS + 2]);

  const int num_warmup_updates = 0;
  const int save_interval = 10;
  ostringstream save_prefix_stream;
  save_prefix_stream << "su" << NCOLORS << "_nd" << NDIMS << "_beta" << beta0
                     << "_L" << lattice_size[0] << "_T"
                     << lattice_size[NDIMS - 1];
  string save_prefix = save_prefix_stream.str();

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

  for (int i = 0; i <= ntraj; ++i) {
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

    // Reunitarize gauge field
    Reunitarize<Real> reu(conf);
    reu.Run();

    // Save gauge configuration with with some interval
    if (i > num_warmup_updates && i % save_interval == 0) {
      string filename = save_prefix + "_cfg_" + to_string(i) + ".bin";
      SaveBin_Gauge<double, double>(conf, filename, false);
    }

    PlaquetteCUB<Real> plaqCUB(conf);
    OnePolyakovLoop<Real> poly(conf);

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

    {
        int rank = 0; // mpirank  
        int gauge_length = NCOLORS * NCOLORS * 4;
        for (int i = 0; i < 4; ++i) {
            gauge_length *= lattice_size[i];
        }
        std::vector<int> mpi_desc{1, 1, 1, 1};
        std::vector<int> global_lattice_desc = lattice_size;
        qcu::io::GaugeStorage<std::complex<double>> gauge_storage(global_lattice_desc, NCOLORS);
        qcu::io::GaugeWriter<double> gauge_writer(rank, mpi_desc);
        int Lx = lattice_size[0];
        int Ly = lattice_size[1];
        int Lz = lattice_size[2];
        int Lt = lattice_size[3];
        // memory allocation
        // Complex<Real>* host_gauge = new Complex<Real>[qcuHeader.GaugeLength()];
        Complex<Real>* device_quda_eo_gauge;
        Complex<Real>* device_quda_gauge;
        CHECK_CUDA(cudaMalloc(&device_quda_eo_gauge, sizeof(Complex<Real>) * gauge_length));
        CHECK_CUDA(cudaMalloc(&device_quda_gauge, sizeof(Complex<Real>) * gauge_length));
        qcu::GaugeEOPreconditioner<Real> preconditioner; // evenodd preconditioner
        // shift sunw to quda 
        Complex<Real>* sunw_ptr = reinterpret_cast<Complex<Real>*>(conf.GetPtr());
        // step1: shift sunw to qio
        qcu::gauge_from_sunwEO_to_qudaEO(device_quda_eo_gauge, sunw_ptr, NCOLORS * NCOLORS, Lx * Ly * Lz * Lt * 4, nullptr);
        // step2: reverse evenodd
        preconditioner.reverse(device_quda_gauge, device_quda_eo_gauge, global_lattice_desc, NCOLORS * NCOLORS, 4, nullptr);
        // step3: copy to host
        CHECK_CUDA(cudaMemcpy (gauge_storage.data_ptr(), device_quda_gauge, sizeof(Complex<Real>) * Lx * Ly * Lz * Lt * 4 * NCOLORS * NCOLORS, cudaMemcpyDeviceToHost));
        // step4: write
        gauge_writer.write("test_gauge.hdf5", gauge_storage);
        // gaugeWriter.write_gauge(reinterpret_cast<complex<Real>*>(host_gauge));
        printf("Gauge writer done\n");
    }


    randstates.Release();
  conf.Release();
  t0.stop();
  COUT << "####################################################################"
       << endl;
  COUT << "Total Time: " << t0.getElapsedTime() << " s" << endl;
  COUT << "####################################################################"
       << endl;

  return;
}
