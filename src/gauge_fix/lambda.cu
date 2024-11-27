
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#ifdef MULTI_GPU
#include <mpi.h>
#endif

#include <comm_mpi.h>
#include <complex.h>
#include <constants.h>
#include <cub/cub.cuh>
#include <cudaAtomic.h>
#include <cuda_common.h>
#include <device_PHB_OVR.h>
#include <device_load_save.h>
#include <exchange.h>
#include <gaugearray.h>
#include <index.h>
#include <matrixsun.h>
#include <reduction.h>
#include <reunitlink.h>
#include <staple.h>
#include <timer.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <lambda.h>

using namespace std;

namespace CULQCD {

////////////////////////////////////////////////////////////////////////////////////////
///  RANDOM GAUGE TRANSFORMATION
////////////////////////////////////////////////////////////////////////////////////////
template <class Real>
__device__ inline msun randomizegx(cuRNGState &localState) {
  msun U;
  for (int i = 0; i < NCOLORS; i++)
    for (int j = 0; j < NCOLORS; j++)
      U.e[i][j] = complex((Real)(Random<Real>(localState) - 0.5),
                          (Real)(Random<Real>(localState) - 0.5));
  return U;
}

template <class Real>
__global__ void kernel_randgaugegx(complex *array, cuRNGState *state) {
  int id = INDEX1D();
  if (id < DEVPARAMS::HalfVolume) {
    cuRNGState localState = state[id];
    for (int oddbit = 0; oddbit < 2; oddbit++) {
      int idd = EOIndeX(id, oddbit);
      msun gx = randomizegx<Real>(localState);
      reunit_link<Real>(&gx);
      GAUGE_SAVE<SOA, Real>(array, gx, idd, DEVPARAMS::VolumeG);
    }
    state[id] = localState;
  }
}

template <bool UseTex, ArrayType atype, class Real>
__global__ void kernel_randgaugetransformation(complex *array, complex *gx) {
  int idd = INDEX1D();
  if (idd < DEVPARAMS::Volume) {
    int oddbit = 0;
    int id = idd;
    if (idd >= DEVPARAMS::HalfVolume) {
      oddbit = 1;
      id -= DEVPARAMS::HalfVolume;
    }
#ifdef MULTI_GPU
    idd = EOIndeX(id, oddbit);
#endif
    msun g = GAUGE_LOAD<false, SOA, Real>(gx, idd, DEVPARAMS::VolumeG);
    for (int nu = 0; nu < 4; nu++) {
      msun U = GAUGE_LOAD<UseTex, atype, Real>(
          array, idd + nu * DEVPARAMS::VolumeG, DEVPARAMS::VolumeG * 4);
      msun U_temp = g * U;
      msun g0 = (GAUGE_LOAD_DAGGER<false, SOA, Real>(
          gx, neighborEOIndexPlusOne(id, oddbit, nu), DEVPARAMS::VolumeG));
      U = U_temp * g0;
      GAUGE_SAVE<atype, Real>(array, U, idd + nu * DEVPARAMS::VolumeG,
                              DEVPARAMS::VolumeG * 4);
    }
  }
}

template <class Real> void RandGaugeTransf(gauge &gaugein, RNG &randstates) {
  if (getVerbosity() >= VERBOSE)
    printfCULQCD("Apply Random Gauge Transformation...\n");
  dim3 threadb(128, 1, 1);
  dim3 blockgx = GetBlockDim(threadb.x, PARAMS::HalfVolume);
  dim3 blockrgt = GetBlockDim(threadb.x, PARAMS::Volume);
  gauge gx(SOA, Device, PARAMS::VolumeG, true, true);
  kernel_randgaugegx<Real>
      <<<blockgx, threadb>>>(gx.GetPtr(), randstates.State());
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    // Only needs to exchange bottom gauge links
    // this function exchange the top and bottom gauge links... leave it for now
    // can be optimized here!!!!!!
    for (int parity = 0; parity < 2; parity++)
      Exchange_gauge_border_links<Real>(gx, 0, parity, true);
  }
#endif
  if (gaugein.EvenOdd()) {
    if (gaugein.Type() == SOA)
      kernel_randgaugetransformation<false, SOA, Real>
          <<<blockrgt, threadb>>>(gaugein.GetPtr(), gx.GetPtr());
    if (gaugein.Type() == SOA12)
      kernel_randgaugetransformation<false, SOA12, Real>
          <<<blockrgt, threadb>>>(gaugein.GetPtr(), gx.GetPtr());
    if (gaugein.Type() == SOA8)
      kernel_randgaugetransformation<false, SOA8, Real>
          <<<blockrgt, threadb>>>(gaugein.GetPtr(), gx.GetPtr());
  } else {
    errorCULQCD("Only defined for even/odd arrays!!!!\n");
  }
  gx.Release();
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    CUDA_SAFE_DEVICE_SYNC();
    // Update border gauge links from node neighbors
    for (int parity = 0; parity < 2; ++parity)
      for (int mu = 0; mu < 4; ++mu) {
        Exchange_gauge_border_links_gauge(gaugein, mu, parity);
      }
  }
#endif
  // CUT_CHECK_ERROR("Error in apply a random gauge transformation...");
  checkCudaError();
  if (getVerbosity() >= VERBOSE)
    printfCULQCD("Applied Random Gauge Transformation...\n");
}
template void RandGaugeTransf<float>(gauges &gaugein, RNG &randstates);
template void RandGaugeTransf<double>(gauged &gaugein, RNG &randstates);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
///  LAMBDA
////////////////////////////////////////////////////////////////////////////////////////
template <class Real> static __device__ inline msun gellman(int g) {
  msun a = msun::zero();
  switch (g) {
  case 0:
    a.e[0][1] = complex::make_complex(1., 0.);
    a.e[1][0] = complex::make_complex(1., 0.);
    break;
  case 1:
    a.e[0][1] = complex::make_complex(0., -1.);
    a.e[1][0] = complex::make_complex(0., 1.);
    break;
  case 2:
    a.e[0][0] = complex::make_complex(1., 0.);
    a.e[1][1] = complex::make_complex(-1., 0.);
    break;
  case 3:
    a.e[0][2] = complex::make_complex(1., 0.);
    a.e[2][0] = complex::make_complex(1., 0.);
    break;
  case 4:
    a.e[0][2] = complex::make_complex(0., -1.);
    a.e[2][0] = complex::make_complex(0., 1.);
    break;
  case 5:
    a.e[1][2] = complex::make_complex(1., 0.);
    a.e[2][1] = complex::make_complex(1., 0.);
    break;
  case 6:
    a.e[1][2] = complex::make_complex(0., -1.);
    a.e[2][1] = complex::make_complex(0., 1.);
    break;
  case 7:
    a.e[0][0] = complex::make_complex(1. / sqrt(3.), 0.);
    a.e[1][1] = complex::make_complex(1. / sqrt(3.), 0.);
    a.e[2][2] = complex::make_complex(-2. / sqrt(3.), 0.);
    break;
  }
  return a * 0.5;
}

template <int blockSize, class Real>
__global__ void kernel_Lambdab(Real *lambdab, Real *meanvalue,
                               cuRNGState *state, Real xi, int size) {
  typedef cub::BlockReduce<Real, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Real res = 0.0;
  int id = INDEX1D();
  if (id < size) {
    int iter = (DEVPARAMS::Volume + size - 1) / size;
    cuRNGState localState = state[id];
    for (int i = 0; i < iter; i++) {
      int idx = id + size * i;
      if (idx >= DEVPARAMS::Volume)
        continue;
      Real tmp = RandomNormal<Real>(localState) * sqrt(xi);
      res += tmp;
      lambdab[idx] = tmp;
    }
    state[id] = localState;
  }
  Real aggregate = BlockReduce(temp_storage).Sum(res);
  if (threadIdx.x == 0)
    CudaAtomicAdd(meanvalue, aggregate);
}

/*
template <typename T>
struct Summ {
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b){
        return a + b;
    }
};*/

template <int blockSize, class Real>
__global__ void kernel_Lambdabb(Real *lambdab, complex *meanvalue,
                                cuRNGState *state, int size) {
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  complex res = complex::zero();
  int id = INDEX1D();
  if (id < size) {
    int iter = (DEVPARAMS::Volume + size - 1) / size;
    cuRNGState localState = state[id];
    for (int i = 0; i < iter; i++) {
      int idx = id + size * i;
      if (idx >= DEVPARAMS::Volume)
        continue;
      Real tmp = RandomNormal<Real>(localState);
      res.real() += tmp;       // for mean value
      res.imag() += tmp * tmp; // for rmsq
      lambdab[idx] = tmp;
    }
    state[id] = localState;
  }
  complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
  if (threadIdx.x == 0)
    CudaAtomicAdd(meanvalue, aggregate);
}

template <int blockSize, class Real>
__global__ void kernel_SetLambdabMean(Real *lambdab, complex *meanvalue,
                                      complex lambdabmean) {
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  complex res = complex::zero();
  int id = INDEX1D();
  if (id < DEVPARAMS::Volume) {
    res.real() = lambdab[id] - lambdabmean.real();
    res.imag() = res.real() * res.real();
    lambdab[id] = res.real();
  }
  complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
  if (threadIdx.x == 0)
    CudaAtomicAdd(meanvalue, aggregate);
}

template <int blockSize, class Real>
__global__ void kernel_LambdabMean(Real *lambdab, complex *meanvalue,
                                   int size) {
  typedef cub::BlockReduce<complex, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  complex res = complex::zero();
  int id = INDEX1D();
  if (id < size) {
    res.real() = lambdab[id];
    res.imag() = res.real() * res.real();
  }

  complex aggregate = BlockReduce(temp_storage).Reduce(res, Summ<complex>());
  if (threadIdx.x == 0)
    CudaAtomicAdd(meanvalue, aggregate);
}
//////////////////////////////////////////////////////////////////
template <class Real>
void CreateLambdaB(Real *lambdab, RNG randstates, string filename,
                   bool binformat, bool savebinsingleprec) {
  Timer makelambdab_time;
  makelambdab_time.start();
  ios::fmtflags old_settings = cout.flags(); // save previous format flags
  int old_precision = cout.precision();      // save previous precision setting
  cout.precision(8);
  cout.setf(ios_base::scientific);
  /////////////////////////////////////////
  bool savetofile = false;
  if (!filename.empty())
    savetofile = true;
  complex *meanvalue = (complex *)dev_malloc(sizeof(complex));
  const int blocksize = 128;
  dim3 threadb(blocksize, 1, 1);
  dim3 block = GetBlockDim(threadb.x, PARAMS::Volume);
  dim3 blockb = GetBlockDim(threadb.x, randstates.Size());
  complex lambdabmean;
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  for (int i = 0; i < 8; i++) {
    COUT << "Generating Lambda_ " << i << endl;
    CUDA_SAFE_CALL(cudaMemset(meanvalue, 0, sizeof(Real)));
    kernel_Lambdabb<blocksize, Real>
        <<<blockb, threadb>>>(lambdab + PARAMS::Volume * i, meanvalue,
                              randstates.State(), randstates.Size());
    CUDA_SAFE_CALL(cudaMemcpy(&lambdabmean, meanvalue, sizeof(complex),
                              cudaMemcpyDeviceToHost));
#ifdef MULTI_GPU
    comm_Allreduce(&lambdabmean);
    lambdabmean /= (Real)numnodes();
#endif
    COUT << "\tSum_x lambda_b(x)/Mean/RMSQ: " << lambdabmean.real();
    lambdabmean /= (Real)PARAMS::Volume;
    COUT << "\t" << lambdabmean.real();
    COUT << "\t" << sqrt(lambdabmean.imag()) << endl;
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaMemset(meanvalue, 0, sizeof(complex)));
    cudaDeviceSynchronize();
    kernel_SetLambdabMean<blocksize, Real><<<block, threadb>>>(
        lambdab + PARAMS::Volume * i, meanvalue, lambdabmean);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaMemcpy(&lambdabmean, meanvalue, sizeof(complex),
                              cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
#ifdef MULTI_GPU
    comm_Allreduce(&lambdabmean);
    lambdabmean /= (Real)numnodes();
#endif
    COUT << "\tSum_x lambda_b(x)/Mean/RMSQ: " << lambdabmean.real();
    lambdabmean /= (Real)PARAMS::Volume;
    COUT << "\t" << lambdabmean.real();
    COUT << "\t" << sqrt(lambdabmean.imag()) << endl;
  }
  block = GetBlockDim(threadb.x, PARAMS::Volume * 8);
  CUDA_SAFE_CALL(cudaMemset(meanvalue, 0, sizeof(complex)));
  kernel_LambdabMean<blocksize, Real>
      <<<block, threadb>>>(lambdab, meanvalue, PARAMS::Volume * 8);
  CUDA_SAFE_CALL(cudaMemcpy(&lambdabmean, meanvalue, sizeof(complex),
                            cudaMemcpyDeviceToHost));
#ifdef MULTI_GPU
  comm_Allreduce(&lambdabmean);
  lambdabmean /= (Real)numnodes();
#endif
  COUT << "Sum_x,b Lambda_b(x)/Mean/RMSQ: " << lambdabmean.real();
  lambdabmean /= (Real)(PARAMS::Volume * 8);
  COUT << "\t" << lambdabmean.real();
  COUT << "\t" << sqrt(lambdabmean.imag()) << endl;
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  if (savetofile) {
    if (savebinsingleprec)
      saveLambdaB<Real, float>(lambdab, Device, filename, binformat);
    else
      saveLambdaB<Real, double>(lambdab, Device, filename, binformat);
  }
  /////////////////////////////////////////
  dev_free(meanvalue);
  cout.flags(old_settings);
  cout.precision(old_precision);
  makelambdab_time.stop();
  COUT << "Time to make LambdaB: " << makelambdab_time.getElapsedTime() << " s"
       << endl;
}
template void CreateLambdaB<float>(float *lambdab, RNG randstates,
                                   string filename, bool binformat,
                                   bool savebinsingleprec);
template void CreateLambdaB<double>(double *lambdab, RNG randstates,
                                    string filename, bool binformat,
                                    bool savebinsingleprec);

//////////////////////////////////////////////////////////////////
template <class Real> struct square_value {
  __device__ Real operator()(const Real &a) const { return a * a; }
};

template <class Real> void Check_LambdaB(Real *lambdab) {
  Timer checklambdab_time;
  checklambdab_time.start();

  ios::fmtflags old_settings = cout.flags(); // save previous format flags
  int old_precision = cout.precision();      // save previous precision setting
  cout.precision(8);
  cout.setf(ios_base::scientific);
  /////////////////////////////////////////
  Real *tmp = (Real *)dev_malloc(sizeof(Real) * 8 * PARAMS::Volume);
  CUDA_SAFE_CALL(cudaMemcpy(tmp, lambdab, sizeof(Real) * 8 * PARAMS::Volume,
                            cudaMemcpyDeviceToDevice));
  Real *tmp_rmsq = (Real *)dev_malloc(sizeof(Real) * 8 * PARAMS::Volume);
  thrust::device_ptr<Real> array = thrust::device_pointer_cast(tmp);
  thrust::device_ptr<Real> array_rmsq = thrust::device_pointer_cast(tmp_rmsq);
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  Real lambdabmean = thrust::reduce(array, array + 8 * PARAMS::Volume);
#ifdef MULTI_GPU
  comm_allreduce(&lambdabmean);
  lambdabmean /= (Real)numnodes();
#endif
  COUT << "Mean/RMSQ/Max/Min: \t" << lambdabmean / (Real)(8 * PARAMS::Volume);
  thrust::transform(array, array + 8 * PARAMS::Volume, array_rmsq,
                    square_value<Real>());
  lambdabmean = thrust::reduce(array_rmsq, array_rmsq + 8 * PARAMS::Volume);
#ifdef MULTI_GPU
  comm_allreduce(&lambdabmean);
  lambdabmean /= (Real)numnodes();
#endif
  COUT << "\t|\t" << sqrt(lambdabmean / (Real)(8 * PARAMS::Volume));
  Real init = 0.0;
  Real result = thrust::reduce(array, array + 8 * PARAMS::Volume, init,
                               thrust::maximum<Real>());
#ifdef MULTI_GPU
  comm_allreduce_max(&result);
#endif
  COUT << "\t|\t" << result;
  init = 0.0;
  result = thrust::reduce(array, array + 8 * PARAMS::Volume, init,
                          thrust::minimum<Real>());
#ifdef MULTI_GPU
  comm_allreduce_min(&result);
#endif
  COUT << "\t|\t" << result << endl;
  for (int i = 0; i < 8; i++) {
    lambdabmean = thrust::reduce(array + i * PARAMS::Volume,
                                 array + (i + 1) * PARAMS::Volume);
#ifdef MULTI_GPU
    comm_allreduce(&lambdabmean);
    lambdabmean /= (Real)numnodes();
#endif
    COUT << i << ":  Mean/RMSQ/Max/Min: \t"
         << lambdabmean / (Real)PARAMS::Volume;
    thrust::transform(array + i * PARAMS::Volume,
                      array + (i + 1) * PARAMS::Volume, array_rmsq,
                      square_value<Real>());
    lambdabmean = thrust::reduce(array_rmsq, array_rmsq + PARAMS::Volume);
#ifdef MULTI_GPU
    comm_allreduce(&lambdabmean);
    lambdabmean /= (Real)numnodes();
#endif
    COUT << "\t|\t" << sqrt(lambdabmean / (Real)(PARAMS::Volume));
    result = thrust::reduce(array + i * PARAMS::Volume,
                            array + (i + 1) * PARAMS::Volume, init,
                            thrust::maximum<Real>());
#ifdef MULTI_GPU
    comm_allreduce_max(&result);
#endif
    COUT << "\t|\t" << result;
    result = thrust::reduce(array + i * PARAMS::Volume,
                            array + (i + 1) * PARAMS::Volume, init,
                            thrust::minimum<Real>());
#ifdef MULTI_GPU
    comm_allreduce_min(&result);
#endif
    COUT << "\t|\t" << result << endl;
  }
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  dev_free(tmp);
  dev_free(tmp_rmsq);
  cout.flags(old_settings);
  cout.precision(old_precision);
  checklambdab_time.stop();
  COUT << "Time to check Lambda: " << checklambdab_time.getElapsedTime() << " s"
       << endl;
}
template void Check_LambdaB<float>(float *lambdab);
template void Check_LambdaB<double>(double *lambdab);

template <class Real, class SavePrec>
void saveLambdaB(Real *Lambdab, ReadMode mode, string filename,
                 bool binformat) {
  Real *ld;
  if (mode == Device) {
    ld = (Real *)safe_malloc(8 * PARAMS::Volume * sizeof(Real));
    CUDA_SAFE_CALL(cudaMemcpy(ld, Lambdab, 8 * PARAMS::Volume * sizeof(Real),
                              cudaMemcpyDeviceToHost));
  } else
    ld = Lambdab;
  ofstream fileout;
  if (mynode() == masternode()) {
    if (binformat) {
      fileout.open(filename.c_str(), ios::binary | ios::out);
      if (!fileout.is_open())
        errorCULQCD("Error saving lambdab to file...\n");
    } else {
      fileout.open(filename.c_str(), ios::out);
      if (!fileout.is_open())
        errorCULQCD("Error saving lambdab to file...\n");
      fileout.precision(14);
      fileout.setf(ios_base::scientific);
    }
  }
  COUT << "Saving lambdab to file " << filename << endl;
  // the lambda_b array is written in odd and even order...
  // save to file in normal order and the 8 numbers are written together...
  SavePrec *tmp = (SavePrec *)safe_malloc(8 * sizeof(SavePrec));
  for (int t = 0; t < PARAMS::NT; t++)
    for (int k = 0; k < PARAMS::NZ; k++)
      for (int j = 0; j < PARAMS::NY; j++)
        for (int i = 0; i < PARAMS::NX; i++) {
#ifdef MULTI_GPU
          int x[4];
          x[0] = i % param_Grid(0);
          x[1] = j % param_Grid(1);
          x[2] = k % param_Grid(2);
          x[3] = t % param_Grid(3);
          int parity = (i + j + k + t) & 1;
          int id = x[0] + x[1] * param_Grid(0) +
                   x[2] * param_Grid(0) * param_Grid(1);
          id += x[3] * param_Grid(0) * param_Grid(1) * param_Grid(2);
          id = id >> 1;
          id += parity * param_HalfVolume();
          int nodetorecv = node_number(i, j, k, t);
          if (mynode() == nodetorecv) {
            for (int l = 0; l < 8; l++)
              tmp[l] = (SavePrec)ld[id + l * PARAMS::Volume];
          }
          if (mynode() == nodetorecv && mynode() != masternode()) {
            MPI_CHECK(MPI_Send(tmp, 8 * sizeof(SavePrec), MPI_BYTE,
                               masternode(), nodetorecv, MPI_COMM_WORLD));
          }
          if (mynode() != nodetorecv && mynode() == masternode()) {
            MPI_CHECK(MPI_Recv(tmp, 8 * sizeof(SavePrec), MPI_BYTE, nodetorecv,
                               nodetorecv, MPI_COMM_WORLD, &MPI_StatuS));
          }
          if (mynode() == masternode()) {
            if (binformat) {
              fileout.write((const char *)tmp, 8 * sizeof(SavePrec));
              if (fileout.fail())
                errorCULQCD("ERROR: Unable save to file...\n");
            } else {
              for (int l = 0; l < 8; l++) {
                fileout << tmp[l] << endl;
                if (fileout.fail())
                  errorCULQCD("ERROR: Unable save to file...\n");
              }
            }
          }
          MPI_Barrier(MPI_COMM_WORLD);
#else
          int idx = i + j * param_Grid(0) + k * param_Grid(0) * param_Grid(1);
          idx += t * param_Grid(0) * param_Grid(1) * param_Grid(2);
          idx = idx >> 1;
          int parity = (i + j + k + t) & 1;
          idx += parity * PARAMS::HalfVolume;
          for (int l = 0; l < 8; l++)
            tmp[l] = (SavePrec)ld[idx + l * PARAMS::Volume];
          if (binformat) {
            fileout.write((const char *)tmp, 8 * sizeof(SavePrec));
            if (fileout.fail())
              errorCULQCD("ERROR: Unable save to file...\n");
          } else {
            for (int l = 0; l < 8; l++) {
              fileout << tmp[l] << endl;
              if (fileout.fail())
                errorCULQCD("ERROR: Unable save to file...\n");
            }
          }
#endif
        }
  host_free(tmp);
  if (mynode() == masternode())
    fileout.close();
  if (mode == Device)
    host_free(ld);
}
template void saveLambdaB<float, float>(float *Lambdab, ReadMode mode,
                                        string filename, bool binformat);
template void saveLambdaB<float, double>(float *Lambdab, ReadMode mode,
                                         string filename, bool binformat);
template void saveLambdaB<double, float>(double *Lambdab, ReadMode mode,
                                         string filename, bool binformat);
template void saveLambdaB<double, double>(double *Lambdab, ReadMode mode,
                                          string filename, bool binformat);

template <class Real, class SavePrec>
void readLambdaB(Real *Lambdab, ReadMode mode, string filename,
                 bool binformat) {
  Real *ld;
  if (mode == Device) {
    ld = (Real *)safe_malloc(8 * PARAMS::Volume * sizeof(Real));
  } else
    ld = Lambdab;

  ifstream filein;
  if (mynode() == masternode()) {
    if (binformat) {
      filein.open(filename.c_str(), ios::binary | ios::in);
      if (!filein.is_open())
        errorCULQCD("Error reading lambdab from file...\n");
    } else {
      filein.open(filename.c_str(), ios::in);
      if (!filein.is_open())
        errorCULQCD("Error reading lambdab from file...\n");
    }
  }
  COUT << "Reading lambdab from file " << filename << endl;
  // the lambda_b array is written in odd and even order...
  SavePrec *tmp = (SavePrec *)safe_malloc(8 * sizeof(SavePrec));
  for (int t = 0; t < PARAMS::NT; t++)
    for (int k = 0; k < PARAMS::NZ; k++)
      for (int j = 0; j < PARAMS::NY; j++)
        for (int i = 0; i < PARAMS::NX; i++) {
#ifdef MULTI_GPU
          int x[4];
          x[0] = i % param_Grid(0);
          x[1] = j % param_Grid(1);
          x[2] = k % param_Grid(2);
          x[3] = t % param_Grid(3);
          int parity = (i + j + k + t) & 1;
          int id = x[0] + x[1] * param_Grid(0) +
                   x[2] * param_Grid(0) * param_Grid(1);
          id += x[3] * param_Grid(0) * param_Grid(1) * param_Grid(2);
          id = id >> 1;
          id += parity * param_HalfVolume();

          if (mynode() == masternode()) {
            if (binformat) {
              filein.read((char *)tmp, 8 * sizeof(SavePrec));
              if (filein.fail())
                errorCULQCD("ERROR: Unable read from file...\n");
            } else {
              for (int l = 0; l < 8; l++) {
                filein >> tmp[l];
                if (filein.fail())
                  errorCULQCD("ERROR: Unable to read from file...\n");
              }
            }
          }
          int nodetosend = node_number(i, j, k, t);
          if (mynode() == nodetosend && mynode() == masternode()) {
            for (int l = 0; l < 8; l++)
              ld[id + l * PARAMS::Volume] = (Real)tmp[l];
          }
          if (mynode() != nodetosend && mynode() == masternode()) {
            MPI_Send(tmp, 8 * sizeof(SavePrec), MPI_BYTE, nodetosend,
                     nodetosend, MPI_COMM_WORLD);
          }
          if (mynode() == nodetosend && mynode() != masternode()) {
            MPI_Recv(tmp, 8 * sizeof(SavePrec), MPI_BYTE, masternode(),
                     mynode(), MPI_COMM_WORLD, &MPI_StatuS);
            for (int l = 0; l < 8; l++)
              ld[id + l * PARAMS::Volume] = (Real)tmp[l];
          }
#else
          int idx = i + j * param_Grid(0) + k * param_Grid(0) * param_Grid(1);
          idx += t * param_Grid(0) * param_Grid(1) * param_Grid(2);
          idx = idx >> 1;
          int parity = (i + j + k + t) & 1;
          idx += parity * PARAMS::HalfVolume;
          if (binformat) {
            filein.read((char *)tmp, 8 * sizeof(SavePrec));
            if (filein.fail())
              errorCULQCD("ERROR: Unable read from file...\n");
          } else {
            for (int l = 0; l < 8; l++) {
              filein >> tmp[l];
              if (filein.fail())
                errorCULQCD("ERROR: Unable to read from file...\n");
            }
          }
          for (int l = 0; l < 8; l++)
            ld[idx + l * PARAMS::Volume] = (Real)tmp[l];
#endif
        }
  host_free(tmp);
  if (mynode() == masternode())
    filein.close();
  if (mode == Device) {
    CUDA_SAFE_CALL(cudaMemcpy(Lambdab, ld, 8 * PARAMS::Volume * sizeof(Real),
                              cudaMemcpyHostToDevice));
    host_free(ld);
  }
}
template void readLambdaB<float, float>(float *Lambdab, ReadMode mode,
                                        string filename, bool binformat);
template void readLambdaB<float, double>(float *Lambdab, ReadMode mode,
                                         string filename, bool binformat);
template void readLambdaB<double, float>(double *Lambdab, ReadMode mode,
                                         string filename, bool binformat);
template void readLambdaB<double, double>(double *Lambdab, ReadMode mode,
                                          string filename, bool binformat);

/*
static inline  __device__ int EOIndeX(int id, int oddbit){
        #ifdef MULTI_GPU
                int x[4];
                getEOCoords3(x, id, DEVPARAMS::Grid, oddbit);
                for(int i=0; i<4;i++) x[i] += param_border(i);
                int idx = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) +
x[1] ) * param_GridG(0) + x[0]) >> 1; if(oddbit) idx += oddbit  *
param_HalfVolumeG(); return idx; #else return id + oddbit *
DEVPARAMS::HalfVolume; #endif
}*/

template <typename T> struct Maxx {
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) {
    if (a > b)
      return a;
    return b;
  }
};
template <> struct Maxx<complexs> {
  __host__ __device__ __forceinline__ complexs operator()(const complexs &a,
                                                          const complexs &b) {
    if (a.abs() > b.abs())
      return a;
    return b;
  }
};
template <> struct Maxx<complexd> {
  __host__ __device__ __forceinline__ complexd operator()(const complexd &a,
                                                          const complexd &b) {
    if (a.abs() > b.abs())
      return a;
    return b;
  }
};

template <class Real, bool mulI, ArrayType atypelambda>
__global__ void kernel_LambdaXi(complex *lambda, Real *lambdab, Real sigma) {
  int id = INDEX1D();
  if (id >= DEVPARAMS::Volume)
    return;
  msun lbda = msun::zero();
  for (int i = 0; i < 8; i++) {
    lbda += gellman<Real>(i) * lambdab[id + DEVPARAMS::Volume * i] * sigma;
  }
  if (mulI)
    lbda *= complex::make_complex(0.0, 1.0);
#ifdef MULTI_GPU
  int parity = 0;
  if (id >= DEVPARAMS::HalfVolume) {
    id -= DEVPARAMS::HalfVolume;
    parity = 1;
  }
  GAUGE_SAVE<atypelambda, Real>(lambda, lbda, EOIndeX(id, parity),
                                DEVPARAMS::VolumeG);
#else
  GAUGE_SAVE<atypelambda, Real>(lambda, lbda, id, DEVPARAMS::Volume);
#endif
}

//////////////////////////////////////////////////////////////////
template <class Real>
void CreateLambdaI(gauge &lambda, Real *lambdab, Real xi) {
  Timer makelambda_time;
  makelambda_time.start();
  // Doing I*Lambda(x) it allows to store using only 12 real number parameters
  if ((lambda.Type() == SOA12 || lambda.Type() == SOA8) && !lambda.EvenOdd())
    errorCULQCD("Error in CreateLambdaI(...) function: lambda can only be a "
                "SOA or SOA12A type in even/odd format...\nExiting...\n");
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  Real sigma = sqrt(2.0 * NCOLORS * xi / (Real)PARAMS::Beta);
  COUT << "Creating \"I * Lambda(x)\" for xi = " << xi
       << " -> sigma = " << sigma << endl;
  const int blocksize = 128;
  dim3 threadb(blocksize, 1, 1);
  dim3 block = GetBlockDim(threadb.x, PARAMS::Volume);
  if (lambda.Type() == SOA)
    kernel_LambdaXi<Real, true, SOA>
        <<<block, threadb>>>(lambda.GetPtr(), lambdab, sigma);
  else if (lambda.Type() == SOA12A)
    kernel_LambdaXi<Real, true, SOA12A>
        <<<block, threadb>>>(lambda.GetPtr(), lambdab, sigma);
  else
    errorCULQCD("Only defined for SOA and SOA12A...\n");
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  complex *tmp = (complex *)dev_malloc(sizeof(complex) * PARAMS::VolumeG);
  thrust::device_ptr<complex> array = thrust::device_pointer_cast(tmp);
  for (int i = 0; i < lambda.getNumElems(); i++) {
    CUDA_SAFE_CALL(cudaMemcpy(tmp, lambda.GetPtr() + i * PARAMS::VolumeG,
                              sizeof(complex) * PARAMS::VolumeG,
                              cudaMemcpyDeviceToDevice));
    complex init = complex::zero();
    complex result =
        thrust::reduce(array, array + PARAMS::VolumeG, init, Maxx<complex>());
#ifdef MULTI_GPU
    comm_Allreduce_Max(&result);
#endif
    COUT << "Max module of elem " << i << ": \t" << result.real() << " + "
         << result.imag() << "I\t|\t abs: " << result.abs() << endl;
  }
  dev_free(tmp);
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  makelambda_time.stop();
  COUT << "Time to make Lambda: " << makelambda_time.getElapsedTime() << " s"
       << endl;
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
}
template void CreateLambdaI<float>(gauges &lambda, float *lambdab, float xi);
template void CreateLambdaI<double>(gauged &lambda, double *lambdab, double xi);

//////////////////////////////////////////////////////////////////
template <class Real> void CreateLambda(gauge &lambda, Real *lambdab, Real xi) {
  Timer makelambda_time;
  makelambda_time.start();
  if (lambda.Type() != SOA && !lambda.EvenOdd())
    errorCULQCD("Error in CreateLambda(...) function: lambda can only be a SOA "
                "in even/odd format...\nExiting...\n");
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  Real sigma = sqrt(2.0 * NCOLORS * xi / (Real)PARAMS::Beta);
  COUT << "Creating Lambda(x) for xi = " << xi << " -> sigma = " << sigma
       << endl;
  const int blocksize = 128;
  dim3 threadb(blocksize, 1, 1);
  dim3 block = GetBlockDim(threadb.x, PARAMS::Volume);
  if (lambda.Type() == SOA)
    kernel_LambdaXi<Real, false, SOA>
        <<<block, threadb>>>(lambda.GetPtr(), lambdab, sigma);
  else
    errorCULQCD("Only defined for SOA...\n");
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  complex *tmp = (complex *)dev_malloc(sizeof(complex) * PARAMS::VolumeG);
  thrust::device_ptr<complex> array = thrust::device_pointer_cast(tmp);
  for (int i = 0; i < lambda.getNumElems(); i++) {
    CUDA_SAFE_CALL(cudaMemcpy(tmp, lambda.GetPtr() + i * PARAMS::VolumeG,
                              sizeof(complex) * PARAMS::VolumeG,
                              cudaMemcpyDeviceToDevice));
    complex init = complex::zero();
    complex result =
        thrust::reduce(array, array + PARAMS::VolumeG, init, Maxx<complex>());
#ifdef MULTI_GPU
    comm_Allreduce_Max(&result);
#endif
    COUT << "Max module of elem " << i << ": \t" << result.real() << " + "
         << result.imag() << "I\t|\t abs: " << result.abs() << endl;
  }
  dev_free(tmp);
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
  makelambda_time.stop();
  COUT << "Time to make Lambda: " << makelambda_time.getElapsedTime() << " s"
       << endl;
  COUT << "--------------------------------------------------------------------"
          "------"
       << endl;
}
template void CreateLambda<float>(gauges &lambda, float *lambdab, float xi);
template void CreateLambda<double>(gauged &lambda, double *lambdab, double xi);

} // namespace CULQCD
