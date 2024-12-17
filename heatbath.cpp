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

#include <iostream>
#include <stdio.h>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()
#include <vector>

// qcu_io
#include "lattice_desc.h"
#include "io/lqcd_read_write.h"
#include "check_error/check_cuda.cuh"
#include "base/datatype/qcu_complex.cuh"
#include "buffer_shape/data_shift.h"
#include "precondition/even_odd_precondition.h"

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
    COUT << "Start generating SU(" << NCOLORS << ") gauge configurations in "
        << NDIMS << " spacetime dimensions" << endl;
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
    if ((argc - 1) != (NDIMS + 5)) {
        errorCULQCD("Number of input arguments is %d, should be (NDIMS + 5) = %d\n",
                    argc - 1, NDIMS + 5);
    }
    vector<int> lattice_size;
    lattice_size.reserve(NDIMS);
    for (int i = 1; i <= NDIMS; i++)
    lattice_size.push_back(atoi(argv[i]));
    double beta0 = atof(argv[NDIMS + 1]);
    PARAMS::UseTex = false;
    int ntraj = atoi(argv[NDIMS + 2]);

    


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
    // SETPARAMS(PARAMS::UseTex, beta0, ns, ns, ns, nt, true);
    SETPARAMS(PARAMS::UseTex, beta0, lattice_size, true);
    gauge conf(mygaugein, Device, PARAMS::Volume * NDIMS, true);
    conf.Details();

    // init the random number generator
    RNG randstates;
    int rngSeed = 1234;
    randstates.Init(rngSeed);

    // Init() for Cold start and Init(randstates) for Hot start
    conf.Init();

    // WJC: my configuration
    if (NDIMS != 4) {
        errorCULQCD("NDIMS != 4, IO only support 4D lattice");
    }
    int save_stride = atoi(argv[NDIMS + 3]);
    std::string gauge_file_suffix = ".bin";
    std::string in_gauge_file = argv[NDIMS + 4]; // "None";
    std::string out_gauge_file_prefix = argv[NDIMS + 5]; // "None";
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "save_stride = " << save_stride 
              << "\nin_gauge_file = " << in_gauge_file
              << "\nout_gauge_file_prefix = " << out_gauge_file_prefix << endl;
    std::cout << "----------------------------------------" << std::endl;

    qcu::FourDimDesc mpi_desc {1, 1, 1, 1};
    qcu::FourDimCoordinate mpi_coordinate {0, 0, 0, 0};
    qcu::FourDimDesc latt_desc {lattice_size[0], lattice_size[1], lattice_size[2], lattice_size[3]};

    std::vector<int> dims_desc{
        NDIMS, 
        latt_desc.data[T_DIM], 
        latt_desc.data[Z_DIM], 
        latt_desc.data[Y_DIM], 
        latt_desc.data[X_DIM], 
        NCOLORS, 
        NCOLORS * 2};
    
    qcu::io::Gauge4Dim<std::complex<Real>> gauge_4d(
        latt_desc.data[T_DIM], 
        latt_desc.data[Z_DIM], 
        latt_desc.data[Y_DIM], 
        latt_desc.data[X_DIM], 
        NCOLORS);

    // memory allocation
    int gauge_length = latt_desc.data[T_DIM] * latt_desc.data[Z_DIM] * latt_desc.data[Y_DIM] * latt_desc.data[X_DIM] 
                        * NCOLORS * NCOLORS * NDIMS;
    std::complex<Real>* host_gauge = reinterpret_cast<std::complex<Real>*>(gauge_4d.data_ptr());
    Complex<Real>* device_quda_eo_gauge;
    Complex<Real>* device_quda_gauge;
    CHECK_CUDA(cudaMalloc(&device_quda_eo_gauge, sizeof(Complex<Real>) * gauge_length));
    CHECK_CUDA(cudaMalloc(&device_quda_gauge, sizeof(Complex<Real>) * gauge_length));
    // memory allocation end

    qcu::GaugeEOPreconditioner<Real> preconditioner; // evenodd preconditioner
    Complex<Real>* sunw_ptr = reinterpret_cast<Complex<Real>*>(conf.GetPtr());

    int mpi_rank = 0;
    if (in_gauge_file != "None") {
        // read gauge
        // GaugeReader<Real> gaugeReader(in_gauge_file, qcuHeader, mpi_desc, mpi_coordinate, latt_desc);
        qcu::io::GaugeReader<Real> gauge_reader{mpi_rank, mpi_desc};
        // step1: read to host
        gauge_reader.read(in_gauge_file, dims_desc, gauge_4d);
        // step2: copy to device
        CHECK_CUDA(cudaMemcpy(device_quda_gauge, host_gauge, sizeof(Complex<Real>) * gauge_length, cudaMemcpyHostToDevice));
        // step3: even-odd precondition
        preconditioner.apply(device_quda_eo_gauge, device_quda_gauge, latt_desc, NCOLORS * NCOLORS, 4, nullptr);
        // step4: shift qio to sunw
        qcu::from_qudaEO_to_sunwEO(
            sunw_ptr, 
            device_quda_eo_gauge, 
            latt_desc.data[T_DIM] * latt_desc.data[Z_DIM] * latt_desc.data[Y_DIM] * latt_desc.data[X_DIM] * 4, 
            NCOLORS * NCOLORS, 
            nullptr);
    }


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

        // if (traj+1) % save_stride == 0, write gauge
        
        if ((i+1) % save_stride == 0 && out_gauge_file_prefix != "None") {
            std::string out_file = out_gauge_file_prefix + "_step_" + std::to_string(i+1) + gauge_file_suffix;
            qcu::io::GaugeWriter<Real> gauge_writer{mpi_rank, mpi_desc};
            // step1: shift sunw to qio
            qcu::gauge_from_sunwEO_to_qudaEO(device_quda_eo_gauge, sunw_ptr, NCOLORS * NCOLORS, latt_desc.data[T_DIM] * latt_desc.data[Z_DIM] * latt_desc.data[Y_DIM] * latt_desc.data[X_DIM] * 4, nullptr);
            // step2: reverse evenodd
            preconditioner.reverse(device_quda_gauge, device_quda_eo_gauge, latt_desc, NCOLORS * NCOLORS, 4, nullptr);
            // step3: copy to host
            CHECK_CUDA(cudaMemcpy (host_gauge, device_quda_gauge, sizeof(Complex<Real>) * gauge_length, cudaMemcpyDeviceToHost));
            // step4: write
            gauge_writer.write(out_file, dims_desc, gauge_4d);
        }
    }

    // last gauge
    {
        if (out_gauge_file_prefix != "None") {
            std::string out_file = out_gauge_file_prefix + gauge_file_suffix;
            qcu::io::GaugeWriter<Real> gauge_writer{mpi_rank, mpi_desc};
            // step1: shift sunw to qio
            qcu::gauge_from_sunwEO_to_qudaEO(device_quda_eo_gauge, sunw_ptr, NCOLORS * NCOLORS, latt_desc.data[T_DIM] * latt_desc.data[Z_DIM] * latt_desc.data[Y_DIM] * latt_desc.data[X_DIM] * 4, nullptr);
            // step2: reverse evenodd
            preconditioner.reverse(device_quda_gauge, device_quda_eo_gauge, latt_desc, NCOLORS * NCOLORS, 4, nullptr);
            // step3: copy to host
            CHECK_CUDA(cudaMemcpy (host_gauge, device_quda_gauge, sizeof(Complex<Real>) * gauge_length, cudaMemcpyDeviceToHost));
            // step4: write
            // gaugeWriter.write_gauge(reinterpret_cast<complex<Real>*>(host_gauge));
            gauge_writer.write(out_file, dims_desc, gauge_4d);
        }
    }

    randstates.Release();

    conf.Release();
    t0.stop();
    COUT << "####################################################################"
        "###########" << endl;
    COUT << "Total Time: " << t0.getElapsedTime() << " s" << endl;
    COUT << "####################################################################"
        "###########" << endl;


    // wjc: free memory
    CHECK_CUDA(cudaFree(device_quda_eo_gauge));
    CHECK_CUDA(cudaFree(device_quda_gauge));
    return;
}
