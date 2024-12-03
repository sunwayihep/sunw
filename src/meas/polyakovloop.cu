
#include <cstdlib>
#include <ctime> // Needed for the true randomization
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <comm_mpi.h>
#include <constants.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <gaugearray.h>
#include <index.h>
#include <meas/polyakovloop.h>
#include <texture_host.h>
#include <timer.h>

#include <reduce_block_1d.h>

using namespace std;

namespace CULQCD {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Onepolyakovloop /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <bool UseTex, ArrayType atypein, ArrayType atypeout, class Real>
__global__ void kernel_calc_polyakovloop_1D(complex *array, complex *ploop) {
  int id = INDEX1D();
  if (id < DEVPARAMS::tstride) {
#ifdef MULTI_GPU
    int index = id + DEVPARAMS::tstride +
                3 * DEVPARAMS::tstride * (DEVPARAMS::Grid[3] + 2);
    int offset = DEVPARAMS::tstride * (DEVPARAMS::Grid[3] + 2) * 4;
#else
    int index = id + (NDIMS - 1) * DEVPARAMS::Volume;
    int offset = DEVPARAMS::Volume * NDIMS;
#endif
    msun L = GAUGE_LOAD<UseTex, atypein, Real>(array, index,
                                               offset); // msun::unit();
    for (int t = 1; t < DEVPARAMS::Grid[NDIMS - 1]; t++)
      L *= GAUGE_LOAD<UseTex, atypein, Real>(
          array, index + t * DEVPARAMS::tstride, offset);
    GAUGE_SAVE<atypeout, Real>(ploop, L, id, DEVPARAMS::tstride);
  }
}

template <bool UseTex, ArrayType atypein, bool savePLMatrix, class Real>
__global__ void kernel_calc_polyakovloop_evenodd(complex *array,
                                                 complex *ploop) {

  uint idd = INDEX1D();
  complex pl = complex::zero();
  if (idd < DEVPARAMS::tstride) {
    int oddbit = 0;
    int id = idd;
    if (idd >= DEVPARAMS::tstride / 2) {
      oddbit = 1;
      id = idd - DEVPARAMS::tstride / 2;
    }
    int x[NDIMS];
    Index_ND_EO(x, id, oddbit);
#ifdef MULTI_GPU
    int idl = (x[0] + x[1] + x[2]);
    for (int i = 0; i < 3; i++)
      x[i] += param_border(i);
    int idx = (((x[2] * param_GridG(1)) + x[1]) * param_GridG(0) + x[0]);
    int mustride = DEVPARAMS::VolumeG;
    int stride = param_GridG(2) * param_GridG(1) * param_GridG(0);
    int offset = mustride * 4;
#else
    int idx = Index_NDs_NM(x);
    int idl = 0;
    for (int i = 0; i < NDIMS - 1; i++)
      idl += x[i];
    int mustride = DEVPARAMS::Volume;
    int stride = 1;
    for (int i = 0; i < NDIMS - 1; i++)
      stride *= param_Grid(i);
    int offset = DEVPARAMS::size;
#endif

    msun L = msun::unit();
    for (int t = 0; t < DEVPARAMS::Grid[NDIMS - 1]; t++) {
#ifdef MULTI_GPU
      int id0 = (idx + (t + param_border(3)) * stride) >> 1;
#else
      int id0 = (idx + t * stride) >> 1;
#endif
      id0 += ((idl + t) & 1) * param_HalfVolumeG();
      L *= GAUGE_LOAD<UseTex, atypein, Real>(
          array, id0 + (NDIMS - 1) * mustride, offset);
    }
    if (savePLMatrix) {
      GAUGE_SAVE<SOA, Real>(ploop, L, idd, DEVPARAMS::tstride);
    } else
      pl = L.trace();
  }
  if (!savePLMatrix)
    reduce_block_1d<complex>(ploop, pl);
}

template <class Real>
void Calculate_OnePloyakovLoop(gauge array, gauge ploop, bool savePLMatrix) {
  uint nthreads = 128;
  dim3 nblocks = GetBlockDim(nthreads, PARAMS::tstride);
  if (ploop.Type() != SOA) {
    cout << "Container to store polyakov loop at each lattice site must be an "
            "SOA type."
         << endl;
  }
  if (savePLMatrix) {
    const bool savematrix = true;
    if (array.EvenOdd()) {
      if (PARAMS::UseTex) {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd<true, SOA, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd<true, SOA12, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd<true, SOA8, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
      } else {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd<false, SOA, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd<false, SOA12, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd<false, SOA8, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
      }
    } else {
      printfError("Not Implemented...\n");
    }
  } else {
    const bool savematrix = false;
    if (array.EvenOdd()) {
      if (PARAMS::UseTex) {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd<true, SOA, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd<true, SOA12, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd<true, SOA8, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
      } else {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd<false, SOA, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd<false, SOA12, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd<false, SOA8, savematrix, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop.GetPtr());
      }
    } else {
      printfError("Not Implemented...\n");
    }
  }
  CUT_CHECK_ERROR("One Polyakov Loop: Kernel execution failed");
  CUDA_SAFE_DEVICE_SYNC();
}

template <bool UseTex, ArrayType atype, class Real>
__global__ void kernel_calc_polyakovloopTR_1D(complex *array, complex *ploop) {
  int id = INDEX1D();
  complex pl = complex::zero();
  if (id < DEVPARAMS::tstride) {
    int index = id + (NDIMS - 1) * DEVPARAMS::Volume;
    msun L = GAUGE_LOAD<UseTex, atype, Real>(array, index);
    for (int t = 1; t < DEVPARAMS::Grid[NDIMS - 1]; t++)
      L *= GAUGE_LOAD<UseTex, atype, Real>(array,
                                           index + t * DEVPARAMS::tstride);
    pl = L.trace();
  }

  reduce_block_1d<complex>(ploop, pl);
}

template <class Real>
void Calculate_TrPloyakovLoop(gauge array, complex *ploop) {
  uint nthreads = 128;
  dim3 nblocks = GetBlockDim(nthreads, PARAMS::tstride);
  if (array.EvenOdd()) {
    if (PARAMS::UseTex) {
      if (array.Type() == SOA)
        kernel_calc_polyakovloop_evenodd<true, SOA, false, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#if (NCOLORS == 3)
      if (array.Type() == SOA12)
        kernel_calc_polyakovloop_evenodd<true, SOA12, false, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
      if (array.Type() == SOA8)
        kernel_calc_polyakovloop_evenodd<true, SOA8, false, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#endif
    } else {
      if (array.Type() == SOA)
        kernel_calc_polyakovloop_evenodd<false, SOA, false, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#if (NCOLORS == 3)
      if (array.Type() == SOA12)
        kernel_calc_polyakovloop_evenodd<false, SOA12, false, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
      if (array.Type() == SOA8)
        kernel_calc_polyakovloop_evenodd<false, SOA8, false, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#endif
    }
  } else {
#ifdef MULTI_GPU
    printfError("Not Implemented...\n");
#else
    if (PARAMS::UseTex) {
      if (array.Type() == SOA)
        kernel_calc_polyakovloopTR_1D<true, SOA, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#if (NCOLORS == 3)
      if (array.Type() == SOA12)
        kernel_calc_polyakovloopTR_1D<true, SOA12, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
      if (array.Type() == SOA8)
        kernel_calc_polyakovloopTR_1D<true, SOA8, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#endif
    } else {
      if (array.Type() == SOA)
        kernel_calc_polyakovloopTR_1D<false, SOA, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#if (NCOLORS == 3)
      if (array.Type() == SOA12)
        kernel_calc_polyakovloopTR_1D<false, SOA12, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
      if (array.Type() == SOA8)
        kernel_calc_polyakovloopTR_1D<false, SOA8, Real>
            <<<nblocks, nthreads, nthreads * sizeof(complex)>>>(array.GetPtr(),
                                                                ploop);
#endif
    }
#endif
  }
  CUT_CHECK_ERROR("Polyakov Loop: Kernel execution failed");
  CUDA_SAFE_DEVICE_SYNC();
}
template void Calculate_TrPloyakovLoop<float>(gauges array, complexs *ploop);
template void Calculate_TrPloyakovLoop<double>(gauged array, complexd *ploop);

template <ArrayType atype, class Real>
__global__ void kernel_ploop_finalmul(complex *array, complex *ploop,
                                      int nodes) {
  int id = INDEX1D();
  complex pl = complex::zero();
  if (id < DEVPARAMS::tstride) {
    msun L = msun::identity();
    int offset = NCOLORS * NCOLORS * DEVPARAMS::tstride;
    for (int i = 0; i < nodes; i++) {
      L *= GAUGE_LOAD<false, atype, Real>(array, id + offset * i,
                                          DEVPARAMS::tstride);
    }
    pl = L.trace();
  }

  reduce_block_1d<complex>(ploop, pl);
}

template <bool UseTex, ArrayType atypein, bool evenoddorder, class Real>
__global__ void kernel_calc_polyakovloop_evenodd00(complex *array,
                                                   msun *ploop) {
  uint idd = INDEX1D();
  if (idd >= DEVPARAMS::tstride)
    return;
  int oddbit = 0;
  int id = idd;
  if (idd >= DEVPARAMS::tstride / 2) {
    oddbit = 1;
    id = idd - DEVPARAMS::tstride / 2;
  }
  int x[NDIMS];
  Index_ND_EO(x, id, oddbit);
#ifdef MULTI_GPU
  for (int i = 0; i < 4; i++)
    x[i] += param_border(i);
  int idx = ((((x[2]) * param_GridG(1)) + x[1]) * param_GridG(0) + x[0]);
  int mustride = DEVPARAMS::VolumeG;
  int stride = param_GridG(2) * param_GridG(1) * param_GridG(0);
  int offset = mustride * 4;
#else
  int idx = Index_NDs_NM(x);
  int mustride = DEVPARAMS::Volume;
  int stride = 1;
  for (int i = 0; i < NDIMS - 1; i++)
    stride *= param_Grid(i);
  int offset = DEVPARAMS::size;
#endif

  msun L = msun::unit();
  int idl = 0;
  for (int i = 0; i < NDIMS; i++)
    idl += x[i];
  for (int t = 0; t < DEVPARAMS::Grid[NDIMS - 1]; t++) {
#ifdef MULTI_GPU
    int id0 = (idx + (t + param_border(3)) * stride) >> 1;
#else
    int id0 = (idx + t * stride) >> 1;
#endif
    id0 += ((idl + t) & 1) * param_HalfVolumeG();
    L *= GAUGE_LOAD<UseTex, atypein, Real>(array, id0 + (NDIMS - 1) * mustride,
                                           offset);
  }
  if (evenoddorder) {
    ploop[idd] = L;
  } else {
#ifdef MULTI_GPU
    idx = ((((x[2]) * param_Grid(1)) + x[1]) * param_Grid(0) + x[0]);
#endif
    ploop[idx] = L;
  }
}

template <class Real>
void Calculate_OnePloyakovLoop(gauge array, msun *ploop, bool evenoddorder) {
  dim3 nthreads(128, 1, 1);
  dim3 nblocks = GetBlockDim(nthreads.x, PARAMS::tstride);
  if (evenoddorder) {
    const bool evenodd = true;
    if (array.EvenOdd()) {
      if (PARAMS::UseTex) {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd00<true, SOA, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd00<true, SOA12, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd00<true, SOA8, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
      } else {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd00<false, SOA, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd00<false, SOA12, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd00<false, SOA8, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
      }
    }
  } else {
    const bool evenodd = false;
    if (array.EvenOdd()) {
      if (PARAMS::UseTex) {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd00<true, SOA, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd00<true, SOA12, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd00<true, SOA8, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
      } else {
        if (array.Type() == SOA)
          kernel_calc_polyakovloop_evenodd00<false, SOA, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA12)
          kernel_calc_polyakovloop_evenodd00<false, SOA12, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
        if (array.Type() == SOA8)
          kernel_calc_polyakovloop_evenodd00<false, SOA8, evenodd, Real>
              <<<nblocks, nthreads>>>(array.GetPtr(), ploop);
      }
    }
  }
  CUT_CHECK_ERROR("One Polyakov Loop: Kernel execution failed");
  CUDA_SAFE_DEVICE_SYNC();
}

template complexs PolyakovLoop<float>(gauges array);
template complexd PolyakovLoop<double>(gauged array);
template <class Real> complex PolyakovLoop(gauge array) {
#ifdef TIMMINGS
  Timer pltime;
  pltime.start();
#endif
  complex *sum = (complex *)dev_malloc(sizeof(complex));
  CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
#ifdef MULTI_GPU
  complex res;
  if (numnodes() == 1) {
    Calculate_TrPloyakovLoop<Real>(array, sum);
    CUDA_SAFE_CALL(
        cudaMemcpy(&res, sum, sizeof(complex), cudaMemcpyDeviceToHost));
    res /= (Real)(NCOLORS * PARAMS::tstride);
  } else if (!comm_dim_partitioned(3)) {
    Calculate_TrPloyakovLoop<Real>(array, sum);
    CUDA_SAFE_CALL(
        cudaMemcpy(&res, sum, sizeof(complex), cudaMemcpyDeviceToHost));
    res /= (Real)(NCOLORS * PARAMS::tstride);
    comm_Allreduce(&res);
    res /= numnodes();
  } else if (comm_dim_partitioned(3) && !comm_dim_partitioned(0) &&
             !comm_dim_partitioned(1) && !comm_dim_partitioned(2)) {
    // NOT OPTIMIZED!!!!!!!!!
    res = complex::zero();
    gauge SUNPloop(SOA, Device, PARAMS::tstride);
    size_t ntocopy = SUNPloop.getNumElems() * PARAMS::tstride;
    complex *ddata;
#ifndef MPI_GPU_DIRECT
    complex *ddata_cpu, *data_cpu_tosend;
#endif
    if (mynode() == masternode()) {
      ddata =
          (complex *)dev_malloc(ntocopy * nodes_per_dim(3) * sizeof(complex));
#ifndef MPI_GPU_DIRECT
      ddata_cpu =
          (complex *)safe_malloc(ntocopy * nodes_per_dim(3) * sizeof(complex));
#endif
    }

    Calculate_OnePloyakovLoop<Real>(array, SUNPloop, true);
#ifndef MPI_GPU_DIRECT
    data_cpu_tosend = (complex *)safe_malloc(ntocopy * sizeof(complex));
    CUDA_SAFE_CALL(cudaMemcpy(data_cpu_tosend, SUNPloop.GetPtr(),
                              ntocopy * sizeof(complex),
                              cudaMemcpyDeviceToHost));
#endif
    CUDA_SAFE_DEVICE_SYNC();
#ifdef MPI_GPU_DIRECT
    MPI_Gather(SUNPloop.GetPtr(), ntocopy, mpi_datatype<complex>(), ddata,
               ntocopy, mpi_datatype<complex>(), masternode(), MPI_COMM_WORLD);
#else
    MPI_Gather(data_cpu_tosend, ntocopy, mpi_datatype<complex>(), ddata_cpu,
               ntocopy, mpi_datatype<complex>(), masternode(), MPI_COMM_WORLD);
    host_free(data_cpu_tosend);
#endif
    SUNPloop.Release();
    if (mynode() == masternode()) {
#ifndef MPI_GPU_DIRECT
      CUDA_SAFE_CALL(cudaMemcpy(ddata, ddata_cpu,
                                ntocopy * nodes_per_dim(3) * sizeof(complex),
                                cudaMemcpyHostToDevice));
#endif
      dim3 threads(128, 1, 1);
      dim3 blocks = GetBlockDim(threads.x, PARAMS::tstride);
      kernel_ploop_finalmul<SOA, Real>
          <<<blocks, threads, threads.x * sizeof(complex)>>>(ddata, sum,
                                                             nodes_per_dim(3));
      CUDA_SAFE_CALL(
          cudaMemcpy(&res, sum, sizeof(complex), cudaMemcpyDeviceToHost));
      res /= (Real)(NCOLORS * PARAMS::tstride);
      dev_free(ddata);
#ifndef MPI_GPU_DIRECT
      host_free(ddata_cpu);
#endif
    }
  } else {
    // NOT OPTIMIZED!!!!!!!!!
    res = complex::zero();
    gauge SUNPloop(SOA, Device, PARAMS::tstride);
    size_t ntocopy = SUNPloop.getNumElems() * PARAMS::tstride;

    Calculate_OnePloyakovLoop<Real>(array, SUNPloop, true);
    CUDA_SAFE_DEVICE_SYNC();

    int disp = PARAMS::logical_coordinate[0] +
               PARAMS::logical_coordinate[1] * nodes_per_dim(0);
    disp += PARAMS::logical_coordinate[2] * nodes_per_dim(0) * nodes_per_dim(1);

    if (mynode() != disp) {
#ifdef MPI_GPU_DIRECT
      MPI_Send(SUNPloop.GetPtr(), ntocopy, mpi_datatype<complex>(), disp,
               mynode(), MPI_COMM_WORLD);
#else
      complex *datatosendcpu =
          (complex *)safe_malloc(ntocopy * sizeof(complex));
      CUDA_SAFE_CALL(cudaMemcpy(datatosendcpu, SUNPloop.GetPtr(),
                                ntocopy * sizeof(complex),
                                cudaMemcpyDeviceToHost));
      MPI_Send(datatosendcpu, ntocopy, mpi_datatype<complex>(), disp, mynode(),
               MPI_COMM_WORLD);
      host_free(datatosendcpu);
#endif
    } else {
      complex *ddata =
          (complex *)dev_malloc(ntocopy * nodes_per_dim(3) * sizeof(complex));
      CUDA_SAFE_CALL(cudaMemcpy(ddata, SUNPloop.GetPtr(),
                                ntocopy * sizeof(complex),
                                cudaMemcpyDeviceToDevice));

      for (int k = 0; k < nodes_per_dim(3) - 1; k++) {
        int fromnode = mynode() + (k + 1) * nodes_per_dim(0) *
                                      nodes_per_dim(1) * nodes_per_dim(2);
#ifdef MPI_GPU_DIRECT
        MPI_Recv(ddata + (k + 1) * ntocopy, ntocopy, mpi_datatype<complex>(),
                 fromnode, fromnode, MPI_COMM_WORLD, &MPI_StatuS);
#else
        complex *datarecv = (complex *)safe_malloc(ntocopy * sizeof(complex));
        MPI_Recv(datarecv, ntocopy, mpi_datatype<complex>(), fromnode, fromnode,
                 MPI_COMM_WORLD, &MPI_StatuS);
        CUDA_SAFE_CALL(cudaMemcpy(ddata + (k + 1) * ntocopy, datarecv,
                                  ntocopy * sizeof(complex),
                                  cudaMemcpyHostToDevice));
        host_free(datarecv);
#endif
      }
      dim3 threads(128, 1, 1);
      dim3 blocks = GetBlockDim(threads.x, PARAMS::tstride);
      kernel_ploop_finalmul<SOA, Real>
          <<<blocks, threads, threads.x * sizeof(complex)>>>(ddata, sum,
                                                             nodes_per_dim(3));
      CUDA_SAFE_CALL(
          cudaMemcpy(&res, sum, sizeof(complex), cudaMemcpyDeviceToHost));
      res /= (Real)(NCOLORS * PARAMS::tstride);
      dev_free(ddata);
    }
    SUNPloop.Release();
    comm_Allreduce(&res);
    res /= (nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2));
  }
#else
  Calculate_TrPloyakovLoop<Real>(array, sum);
  complex res;
  CUDA_SAFE_CALL(
      cudaMemcpy(&res, sum, sizeof(complex), cudaMemcpyDeviceToHost));
  res /= (Real)(NCOLORS * PARAMS::tstride);
#endif
#ifdef TIMMINGS
  CUDA_SAFE_DEVICE_SYNC();
  CUT_CHECK_ERROR("Kernel execution failed");
  printfCULQCD("Polyakov Loop: < %.12e : %.12e : %.12e >\n", res.real(),
               res.imag(), res.abs());
  pltime.stop();
  // NEED TO RECOUNT SINCE FLOPS will be DEPENDING ON HOW LATTICE IS
  // PARTITIONED!!!!!
  long long size_space = 1;
  for (int i = 0; i < NDIMS - 1; i++)
    size_space *= PARAMS::Grid[i];
  long long ThreadReadWrites =
      size_space * (array.getNumParams() * PARAMS::Grid[NDIMS - 1] + 2LL);
#if (NCOLORS == 3)
  long long ThreadFlop = (4LL + 198. * PARAMS::Grid[NDIMS - 1]) * size_space;
#else
  long long ThreadFlop =
      ((NCOLORS - 1) * 2LL +
       NCOLORS * NCOLORS * NCOLORS * 8LL * PARAMS::Grid[NDIMS - 1]) *
      size_space;
#endif

  // NEED TO RECOUNT!!!!!!

#ifdef MULTI_GPU
  // THIS WAS NOT CHECKED YET..... STILL HAVE TO IMPROVE PERFORMANCE...
  float TotalGBytes = numnodes() * float(ThreadReadWrites) * sizeof(Real) /
                      (pltime.getElapsedTimeInSec() * (float)(1 << 30));
  float TotalGFlops =
      numnodes() * (float(ThreadFlop) * 1.0e-9) / pltime.getElapsedTimeInSec();
  COUT << "Polyakov Loop:  " << pltime.getElapsedTimeInSec() << " s\t"
       << TotalGBytes << " GB/s\t" << TotalGFlops << " GFlops" << endl;
#else
  float TotalGBytes = float(ThreadReadWrites) * sizeof(Real) /
                      (pltime.getElapsedTimeInSec() * (float)(1 << 30));
  float TotalGFlops =
      (float(ThreadFlop) * 1.0e-9) / pltime.getElapsedTimeInSec();
  cout << "Polyakov Loop:  " << pltime.getElapsedTimeInSec() << " s\t"
       << TotalGBytes << " GB/s\t" << TotalGFlops << " GFlops" << endl;
#endif
#endif
  dev_free(sum);
  return res;
}

template <class Real>
OnePolyakovLoop<Real>::OnePolyakovLoop(gauge &array) : array(array) {
  if (!array.EvenOdd())
    errorCULQCD("Only implemented for even/odd array...");
  functionName = "Polyakov Loop";
  poly_value = complex::zero();
  size = 1;
  for (int i = 0; i < NDIMS; i++)
    grid[i] = PARAMS::Grid[i];
  size = 1;
  for (int i = 0; i < NDIMS - 1; i++)
    size *= PARAMS::Grid[i];
  timesec = 0.0;
  SetFunctionPtr();
  sum = (complex *)dev_malloc(sizeof(complex));
}

template <class Real> OnePolyakovLoop<Real>::~OnePolyakovLoop() {
  if (sum) {
    dev_free(sum);
    sum = 0;
  }
}

template <class Real> void OnePolyakovLoop<Real>::SetFunctionPtr() {
  tex = PARAMS::UseTex;
  kernel_pointer = NULL;
  tmp = NULL;
#ifdef MULTI_GPU
  if (numnodes() == 1 || !comm_dim_partitioned(3)) {
    if (array.EvenOdd()) {
      if (tex) {
#if (NCOLORS == 3)
        if (array.Type() == SOA)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<true, SOA, false, Real>;
        if (array.Type() == SOA12)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<true, SOA12, false, Real>;
        if (array.Type() == SOA8)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<true, SOA8, false, Real>;
#else
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<true, SOA, false, Real>;
#endif
      } else {
#if (NCOLORS == 3)
        if (array.Type() == SOA)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<false, SOA, false, Real>;
        if (array.Type() == SOA12)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<false, SOA12, false, Real>;
        if (array.Type() == SOA8)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<false, SOA8, false, Real>;
#else
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<false, SOA, false, Real>;
#endif
      }
    }
  } else {
    if (array.EvenOdd()) {
      if (tex) {
#if (NCOLORS == 3)
        if (array.Type() == SOA)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<true, SOA, true, Real>;
        if (array.Type() == SOA12)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<true, SOA12, true, Real>;
        if (array.Type() == SOA8)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<true, SOA8, true, Real>;
#else
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<true, SOA, true, Real>;
#endif
      } else {
#if (NCOLORS == 3)
        if (array.Type() == SOA)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<false, SOA, true, Real>;
        if (array.Type() == SOA12)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<false, SOA12, true, Real>;
        if (array.Type() == SOA8)
          kernel_pointer =
              &kernel_calc_polyakovloop_evenodd<false, SOA8, true, Real>;
#else
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<false, SOA, true, Real>;
#endif
      }
    }
  }
#else
  if (array.EvenOdd()) {
    if (tex) {
#if (NCOLORS == 3)
      if (array.Type() == SOA)
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<true, SOA, false, Real>;
      if (array.Type() == SOA12)
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<true, SOA12, false, Real>;
      if (array.Type() == SOA8)
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<true, SOA8, false, Real>;
#else
      kernel_pointer =
          &kernel_calc_polyakovloop_evenodd<true, SOA, false, Real>;
#endif
    } else {
#if (NCOLORS == 3)
      if (array.Type() == SOA)
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<false, SOA, false, Real>;
      if (array.Type() == SOA12)
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<false, SOA12, false, Real>;
      if (array.Type() == SOA8)
        kernel_pointer =
            &kernel_calc_polyakovloop_evenodd<false, SOA8, false, Real>;
#else
      kernel_pointer =
          &kernel_calc_polyakovloop_evenodd<false, SOA, false, Real>;
#endif
    }
  }
#endif
  if (kernel_pointer == NULL)
    errorCULQCD(
        "No kernel Polyakov Loop function exist for this gauge array...");
}

template <class Real>
void OnePolyakovLoop<Real>::apply(const cudaStream_t &stream) {
  TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef MULTI_GPU
  if (numnodes() == 1 || !comm_dim_partitioned(3))
    kernel_pointer<<<tp.grid, tp.block, tp.block.x * sizeof(complex), stream>>>(
        array.GetPtr(), sum);
  else {
    kernel_pointer<<<tp.grid, tp.block, 0, stream>>>(array.GetPtr(), tmp);
  }
#else
  kernel_pointer<<<tp.grid, tp.block, tp.block.x * sizeof(complex), stream>>>(
      array.GetPtr(), sum);
#endif
}
template <class Real>
complex OnePolyakovLoop<Real>::Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
  mtime.start();
#endif
  // just ensure that the texture was not unbind somewhere...
  if (tex != PARAMS::UseTex) {
    SetFunctionPtr();
  }
  GAUGE_TEXTURE(array.GetPtr(), true);
#ifdef MULTI_GPU
  if (numnodes() == 1) {
    CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
    apply(stream);
    CUDA_SAFE_CALL(
        cudaMemcpy(&poly_value, sum, sizeof(complex), cudaMemcpyDeviceToHost));
    poly_value /= (Real)(NCOLORS * size);
  } else if (!comm_dim_partitioned(3)) {
    CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
    apply(stream);
    CUDA_SAFE_CALL(
        cudaMemcpy(&poly_value, sum, sizeof(complex), cudaMemcpyDeviceToHost));
    poly_value /= (Real)(NCOLORS * size);
    comm_Allreduce(&poly_value);
    poly_value /= numnodes();
  } else if (comm_dim_partitioned(3) && !comm_dim_partitioned(0) &&
             !comm_dim_partitioned(1) && !comm_dim_partitioned(2)) {
    // NOT OPTIMIZED!!!!!!!!!
    poly_value = complex::zero();
    tmp = (complex *)dev_malloc(size * sizeof(complex) * NCOLORS * NCOLORS);
    apply(stream);

    size_t ntocopy = NCOLORS * NCOLORS * PARAMS::tstride * sizeof(complex);
    complex *ddata;
#ifndef MPI_GPU_DIRECT
    complex *ddata_cpu, *data_cpu_tosend;
#endif
    if (mynode() == masternode()) {
      ddata = (complex *)dev_malloc(ntocopy * nodes_per_dim(3));
#ifndef MPI_GPU_DIRECT
      ddata_cpu = (complex *)safe_malloc(ntocopy * nodes_per_dim(3));
#endif
    }
#ifndef MPI_GPU_DIRECT
    data_cpu_tosend = (complex *)safe_malloc(ntocopy);
    CUDA_SAFE_CALL(
        cudaMemcpy(data_cpu_tosend, tmp, ntocopy, cudaMemcpyDeviceToHost));
#endif
    CUDA_SAFE_DEVICE_SYNC();
#ifdef MPI_GPU_DIRECT
    MPI_Gather(tmp, ntocopy, MPI_BYTE, ddata, ntocopy, MPI_BYTE, masternode(),
               MPI_COMM_WORLD);
#else
    MPI_Gather(data_cpu_tosend, ntocopy, MPI_BYTE, ddata_cpu, ntocopy, MPI_BYTE,
               masternode(), MPI_COMM_WORLD);
    host_free(data_cpu_tosend);
#endif
    dev_free(tmp);
    if (mynode() == masternode()) {
#ifndef MPI_GPU_DIRECT
      CUDA_SAFE_CALL(cudaMemcpy(ddata, ddata_cpu, ntocopy * nodes_per_dim(3),
                                cudaMemcpyHostToDevice));
#endif
      dim3 threads(128, 1, 1);
      dim3 blocks = GetBlockDim(threads.x, PARAMS::tstride);
      CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
      kernel_ploop_finalmul<SOA, Real>
          <<<blocks, threads, threads.x * sizeof(complex), stream>>>(
              ddata, sum, nodes_per_dim(3));
      CUDA_SAFE_CALL(cudaMemcpy(&poly_value, sum, sizeof(complex),
                                cudaMemcpyDeviceToHost));
      poly_value /= (Real)(NCOLORS * size);
      dev_free(ddata);
#ifndef MPI_GPU_DIRECT
      host_free(ddata_cpu);
#endif
    }
    CUDA_SAFE_DEVICE_SYNC();
  } else {
    // NOT OPTIMIZED!!!!!!!!!
    poly_value = complex::zero();
    tmp = (complex *)dev_malloc(size * sizeof(complex) * NCOLORS * NCOLORS);
    size_t ntocopy = NCOLORS * NCOLORS * PARAMS::tstride;

    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();

    int disp = PARAMS::logical_coordinate[0] +
               PARAMS::logical_coordinate[1] * nodes_per_dim(0);
    disp += PARAMS::logical_coordinate[2] * nodes_per_dim(0) * nodes_per_dim(1);

    if (mynode() != disp) {
#ifdef MPI_GPU_DIRECT
      MPI_Send(tmp, ntocopy, mpi_datatype<complex>(), disp, mynode(),
               MPI_COMM_WORLD);
#else
      complex *datatosendcpu =
          (complex *)safe_malloc(ntocopy * sizeof(complex));
      CUDA_SAFE_CALL(cudaMemcpy(datatosendcpu, tmp, ntocopy * sizeof(complex),
                                cudaMemcpyDeviceToHost));
      MPI_Send(datatosendcpu, ntocopy, mpi_datatype<complex>(), disp, mynode(),
               MPI_COMM_WORLD);
      host_free(datatosendcpu);
#endif
    } else {
      complex *ddata =
          (complex *)dev_malloc(ntocopy * nodes_per_dim(3) * sizeof(complex));
      CUDA_SAFE_CALL(cudaMemcpy(ddata, tmp, ntocopy * sizeof(complex),
                                cudaMemcpyDeviceToDevice));

      for (int k = 0; k < nodes_per_dim(3) - 1; k++) {
        int fromnode = mynode() + (k + 1) * nodes_per_dim(0) *
                                      nodes_per_dim(1) * nodes_per_dim(2);
#ifdef MPI_GPU_DIRECT
        MPI_Recv(ddata + (k + 1) * ntocopy, ntocopy, mpi_datatype<complex>(),
                 fromnode, fromnode, MPI_COMM_WORLD, &MPI_StatuS);
#else
        complex *datarecv = (complex *)safe_malloc(ntocopy * sizeof(complex));
        MPI_Recv(datarecv, ntocopy, mpi_datatype<complex>(), fromnode, fromnode,
                 MPI_COMM_WORLD, &MPI_StatuS);
        CUDA_SAFE_CALL(cudaMemcpy(ddata + (k + 1) * ntocopy, datarecv,
                                  ntocopy * sizeof(complex),
                                  cudaMemcpyHostToDevice));
        host_free(datarecv);
#endif
      }
      dim3 threads(128, 1, 1);
      dim3 blocks = GetBlockDim(threads.x, PARAMS::tstride);
      CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
      kernel_ploop_finalmul<SOA, Real>
          <<<blocks, threads, threads.x * sizeof(complex), stream>>>(
              ddata, sum, nodes_per_dim(3));
      CUDA_SAFE_CALL(cudaMemcpy(&poly_value, sum, sizeof(complex),
                                cudaMemcpyDeviceToHost));
      poly_value /= (Real)(NCOLORS * size);
      dev_free(ddata);
    }
    dev_free(tmp);
    comm_Allreduce(&poly_value);
    poly_value /= (nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2));
  }
#else
  CUDA_SAFE_CALL(cudaMemset(sum, 0, sizeof(complex)));
  apply(stream);
  CUDA_SAFE_CALL(
      cudaMemcpy(&poly_value, sum, sizeof(complex), cudaMemcpyDeviceToHost));
  poly_value /= (Real)(NCOLORS * size);
#endif
#ifdef TIMMINGS
  CUDA_SAFE_DEVICE_SYNC();
  mtime.stop();
  timesec = mtime.getElapsedTimeInSec();
#endif
  return poly_value;
}
template <class Real> complex OnePolyakovLoop<Real>::Run() { return Run(0); }

template <class Real> double OnePolyakovLoop<Real>::time() { return timesec; }

template <class Real> void OnePolyakovLoop<Real>::stat() {
  COUT << "Polyakov Loop:  " << time() << " s\t" << bandwidth() << " GB/s\t"
       << flops() << " GFlops" << endl;
}

template <class Real> void OnePolyakovLoop<Real>::printValue() {
  printfCULQCD("Polyakov Loop: < %.12e : %.12e : %.12e >\n", poly_value.real(),
               poly_value.imag(), poly_value.abs());
}

template <class Real> long long OnePolyakovLoop<Real>::flop() const {
  size_t spacial_volume = 1;
  for (int i = 0; i < NDIMS - 1; i++)
    spacial_volume *= grid[i];

#if (NCOLORS == 3)
  return (4LL + 198LL * grid[NDIMS - 1]) * spacial_volume * numnodes();
#else
  return ((NCOLORS - 1) * 2LL +
          NCOLORS * NCOLORS * NCOLORS * 8LL * grid[NDIMS - 1]) *
         spacial_volume * numnodes();
#endif
}
template <class Real> long long OnePolyakovLoop<Real>::bytes() const {
  size_t spacial_volume = 1;
  for (int i = 0; i < NDIMS - 1; i++)
    spacial_volume *= grid[i];
  return spacial_volume * (array.getNumParams() * grid[NDIMS - 1] + 2LL) *
         numnodes() * sizeof(Real);
}

template <class Real> double OnePolyakovLoop<Real>::flops() {
  return ((double)flop() * 1.0e-9) / timesec;
}
template <class Real> double OnePolyakovLoop<Real>::bandwidth() {
  return (double)bytes() / (timesec * (double)(1 << 30));
}

template class OnePolyakovLoop<float>;
template class OnePolyakovLoop<double>;

} // namespace CULQCD
