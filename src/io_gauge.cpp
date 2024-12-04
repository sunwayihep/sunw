
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <comm_mpi.h>
#include <complex.h>
#include <constants.h>
#include <cuda_common.h>
#include <exchange.h>
#include <gaugearray.h>
#include <matrixsun.h>

#include <io_gauge.h>

using namespace std;

namespace CULQCD {

template void Read_GaugeOLD<float>(gauges arrayin, std::string filename);
template void Read_GaugeOLD<double>(gauged arrayin, std::string filename);

template <class Real> void Read_GaugeOLD(gauge arrayin, std::string filename) {
  if (!arrayin.EvenOdd())
    errorCULQCD("Not implemented reading a non even/odd array format...\n");

  gauge array = arrayin;
  if (arrayin.Mode() == Device) {
    gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
              arrayin.Border());
    array = A_h;
  }
  ifstream filein;
  filein.open(filename.c_str(), ios::in);
  if (filein.is_open()) {
    int line = 0;
    COUT << "Reading configuration " << filename << endl;
    string header;
    getline(filein, header);
    COUT << header << endl;
    /*for(int t=0; t< param_Grid(3);t++)
    for(int k=0; k< param_Grid(2);k++)
    for(int j=0; j< param_Grid(1);j++)
    for(int i=0; i< param_Grid(0);i++)*/
    for (int i = 0; i < param_Grid(0); i++)
      for (int j = 0; j < param_Grid(1); j++)
        for (int k = 0; k < param_Grid(2); k++)
          for (int t = 0; t < param_Grid(3); t++) {
            int parity = (i + j + k + t) & 1;
            int id = i + j * param_Grid(0) + k * param_Grid(0) * param_Grid(1);
            id += t * param_Grid(0) * param_Grid(1) * param_Grid(2);
            id = id >> 1;
            id += parity * param_HalfVolume();
            for (int dir = 0; dir < 4; dir++) {
              int idx = id + dir * param_Volume();
              msun m;
              for (int i1 = 0; i1 < NCOLORS; i1++)
                for (int j1 = 0; j1 < NCOLORS; j1++) {
                  line++;
                  int pos = idx + (j1 + i1 * NCOLORS) * 4 * param_Volume();
                  filein >> m.e[i1][j1].real() >> m.e[i1][j1].imag();
                  if (filein.fail()) {
                    errorCULQCD("ERROR: Unable to read file: %s in line %d\n",
                                filename.c_str(), line);
                  }
                }
              array.Set(m, idx);
            }
          }
    COUT << "Finisning reading configuration..." << endl;
  } else {
    errorCULQCD("Error reading configuration: %s\n", filename.c_str());
  }
  filein.close();
  if (arrayin.Mode() == Device) {
    CUDA_SAFE_CALL(
        cudaMemcpy(arrayin.GetPtr(), array.GetPtr(),
                   sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                   cudaMemcpyHostToDevice));
    array.Release();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void Save_Gauge<float>(gauges arrayin, string filename);
template void Save_Gauge<double>(gauged arrayin, string filename);

template <class Real> void Save_Gauge(gauge arrayin, string filename) {
  if (!arrayin.EvenOdd())
    errorCULQCD("Not implemented saving a non even/odd array format...\n");
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    if (!arrayin.Border())
      errorCULQCD("Not implemented saving a non gauge array format without "
                  "border in multi-GPU mode...\n");
    gauge array = arrayin;
    if (arrayin.Mode() == Device) {
      gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
                arrayin.Border());
      CUDA_SAFE_CALL(
          cudaMemcpy(A_h.GetPtr(), arrayin.GetPtr(),
                     sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                     cudaMemcpyDeviceToHost));
      array = A_h;
    }
    ofstream fileout;
    if (mynode() == masternode()) {
      fileout.open(filename.c_str(), ios::out);
      fileout.precision(14);
      fileout.setf(ios_base::scientific);
      if (!fileout.is_open()) {
        errorCULQCD("Error saving configuration: %s\n", filename.c_str());
      }
    }
    int line = 0;
    COUT << "Saving configuration " << filename << endl;
    msun *data = (msun *)safe_malloc(sizeof(msun) * 4);
    for (int t = 0; t < PARAMS::NT; t++)
      for (int k = 0; k < PARAMS::NZ; k++)
        for (int j = 0; j < PARAMS::NY; j++)
          for (int i = 0; i < PARAMS::NX; i++) {
            int x[4];
            x[0] = i % param_Grid(0);
            x[1] = j % param_Grid(1);
            x[2] = k % param_Grid(2);
            x[3] = t % param_Grid(3);
            for (int id = 0; id < 4; id++) {
              x[id] += param_border(id);
            }
            int parity = (i + j + k + t) & 1;
            int id = x[0] + x[1] * param_GridG(0) +
                     x[2] * param_GridG(0) * param_GridG(1);
            id += x[3] * param_GridG(0) * param_GridG(1) * param_GridG(2);
            id = id >> 1;
            id += parity * param_HalfVolumeG();

            int nodetorecv = node_number(i, j, k, t);
            if (mynode() == nodetorecv) {
              for (int dir = 0; dir < 4; dir++) {
                int idx = id + dir * param_VolumeG();
                data[dir] = array.Get(idx);
              }
            }
            if (mynode() == nodetorecv && mynode() != masternode()) {
              MPI_Send(data, NCOLORS * NCOLORS * 4, mpi_datatype<complex>(),
                       masternode(), mynode(), MPI_COMM_WORLD);
            }
            if (mynode() != nodetorecv && mynode() == masternode()) {
              MPI_Recv(data, NCOLORS * NCOLORS * 4, mpi_datatype<complex>(),
                       nodetorecv, nodetorecv, MPI_COMM_WORLD, &MPI_StatuS);
            }
            if (mynode() == masternode()) {
              for (int dir = 0; dir < 4; dir++) {
                for (int i1 = 0; i1 < NCOLORS; i1++)
                  for (int j1 = 0; j1 < NCOLORS; j1++) {
                    line++;
                    fileout << data[dir].e[i1][j1].real() << "\t"
                            << data[dir].e[i1][j1].imag() << endl;
                    if (fileout.fail()) {
                      errorCULQCD("ERROR: Unable save to file: %s in line %d\n",
                                  filename.c_str(), line);
                    }
                  }
              }
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
    host_free(data);
    COUT << "Finisning saving configuration..." << endl;
    if (mynode() == masternode())
      fileout.close();
    if (arrayin.Mode() == Device) {
      array.Release();
    }
    return;
  }
#endif
  gauge array = arrayin;
  if (arrayin.Mode() == Device) {
    gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
              arrayin.Border());
    CUDA_SAFE_CALL(
        cudaMemcpy(A_h.GetPtr(), arrayin.GetPtr(),
                   sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                   cudaMemcpyDeviceToHost));
    array = A_h;
  }
  int line = 0;
  ofstream fileout;
  fileout.open(filename.c_str(), ios::out);
  fileout.precision(14);
  fileout.setf(ios_base::scientific);
  if (fileout.is_open()) {
    cout << "Saving configuration " << filename << endl;
    for (int t = 0; t < param_Grid(3); t++)
      for (int k = 0; k < param_Grid(2); k++)
        for (int j = 0; j < param_Grid(1); j++)
          for (int i = 0; i < param_Grid(0); i++) {
            int parity = (i + j + k + t) & 1;
            int id = i + j * param_Grid(0) + k * param_Grid(0) * param_Grid(1);
            id += t * param_Grid(0) * param_Grid(1) * param_Grid(2);
            id = id >> 1;
            id += parity * param_HalfVolume();
            for (int dir = 0; dir < 4; dir++) {
              int idx = id + dir * param_Volume();
              msun m = array.Get(idx);
              for (int i1 = 0; i1 < NCOLORS; i1++)
                for (int j1 = 0; j1 < NCOLORS; j1++) {
                  line++;
                  fileout << m.e[i1][j1].real() << "\t" << m.e[i1][j1].imag()
                          << endl;
                  if (fileout.fail()) {
                    errorCULQCD("ERROR: Unable save to file: %s in line %d\n",
                                filename.c_str(), line);
                  }
                }
            }
          }
    COUT << "Finisning saving configuration..." << endl;
  } else {
    errorCULQCD("Error saving configuration: %s\n", filename.c_str());
  }
  fileout.close();
  if (arrayin.Mode() == Device)
    array.Release();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void Read_Gauge<float>(gauges arrayin, std::string filename);
template void Read_Gauge<double>(gauged arrayin, std::string filename);

template <class Real> void Read_Gauge(gauge arrayin, std::string filename) {
  if (!arrayin.EvenOdd())
    errorCULQCD("Not implemented reading a non even/odd array format...\n");
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    if (!arrayin.Border())
      errorCULQCD("Not implemented reading a non gauge array format without "
                  "border in multi-GPU mode...\n");
    gauge array = arrayin;
    if (arrayin.Mode() == Device) {
      gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
                arrayin.Border());
      array = A_h;
    }
    ifstream filein;
    if (mynode() == masternode()) {
      filein.open(filename.c_str(), ios::in);
      if (!filein.is_open()) {
        errorCULQCD("Error reading configuration: %s\n", filename.c_str());
      }
    }
    int line = 0;
    COUT << "Reading configuration " << filename << endl;
    for (int t = 0; t < PARAMS::NT; t++)
      for (int k = 0; k < PARAMS::NZ; k++)
        for (int j = 0; j < PARAMS::NY; j++)
          for (int i = 0; i < PARAMS::NX; i++) {
            int x[4];
            x[0] = i % param_Grid(0);
            x[1] = j % param_Grid(1);
            x[2] = k % param_Grid(2);
            x[3] = t % param_Grid(3);
            for (int id = 0; id < 4; id++) {
              x[id] += param_border(id);
            }
            int parity = (i + j + k + t) & 1;
            int id = x[0] + x[1] * param_GridG(0) +
                     x[2] * param_GridG(0) * param_GridG(1);
            id += x[3] * param_GridG(0) * param_GridG(1) * param_GridG(2);
            id = id >> 1;
            id += parity * param_HalfVolumeG();
            for (int dir = 0; dir < 4; dir++) {
              int idx = id + dir * param_VolumeG();
              msun m;
              if (mynode() == masternode()) {
                for (int i1 = 0; i1 < NCOLORS; i1++)
                  for (int j1 = 0; j1 < NCOLORS; j1++) {
                    line++;
                    filein >> m.e[i1][j1].real() >> m.e[i1][j1].imag();
                    if (filein.fail()) {
                      errorCULQCD("ERROR: Unable to read file: %s in line %d\n",
                                  filename.c_str(), line);
                    }
                  }
              }
              int nodetosend = node_number(i, j, k, t);
              if (mynode() == nodetosend && mynode() == masternode()) {
                array.Set(m, idx);
              }
              if (mynode() != nodetosend && mynode() == masternode()) {
                MPI_Send(&m, NCOLORS * NCOLORS, mpi_datatype<complex>(),
                         nodetosend, nodetosend, MPI_COMM_WORLD);
              }
              if (mynode() == nodetosend && mynode() != masternode()) {
                msun mm;
                MPI_Recv(&mm, NCOLORS * NCOLORS, mpi_datatype<complex>(),
                         masternode(), mynode(), MPI_COMM_WORLD, &MPI_StatuS);
                array.Set(mm, idx);
              }
            }
          }
    COUT << "Finisning reading configuration..." << endl;
    if (mynode() == masternode())
      filein.close();
    if (arrayin.Mode() == Device) {
      CUDA_SAFE_CALL(cudaMemcpy(arrayin.GetPtr(), array.GetPtr(),
                                arrayin.Bytes(), cudaMemcpyHostToDevice));
      array.Release();
    }
    CUDA_SAFE_DEVICE_SYNC();
    for (int parity = 0; parity < 2; ++parity)
      for (int mu = 0; mu < 4; ++mu) {
        Exchange_gauge_border_links_gauge(arrayin, mu, parity);
      }
    return;
  }
#endif
  gauge array = arrayin;
  if (arrayin.Mode() == Device) {
    gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
              arrayin.Border());
    array = A_h;
  }
  ifstream filein;
  filein.open(filename.c_str(), ios::in);
  if (filein.is_open()) {
    int line = 0;
    cout << "Reading configuration " << filename << endl;
    for (int t = 0; t < param_Grid(3); t++)
      for (int k = 0; k < param_Grid(2); k++)
        for (int j = 0; j < param_Grid(1); j++)
          for (int i = 0; i < param_Grid(0); i++) {
            int parity = (i + j + k + t) & 1;
            int id = i + j * param_Grid(0) + k * param_Grid(0) * param_Grid(1);
            id += t * param_Grid(0) * param_Grid(1) * param_Grid(2);
            id = id >> 1;
            id += parity * param_HalfVolume();
            for (int dir = 0; dir < 4; dir++) {
              int idx = id + dir * param_Volume();
              msun m;
              for (int i1 = 0; i1 < NCOLORS; i1++)
                for (int j1 = 0; j1 < NCOLORS; j1++) {
                  line++;
                  int pos = idx + (j1 + i1 * NCOLORS) * 4 * param_Volume();
                  filein >> m.e[i1][j1].real() >> m.e[i1][j1].imag();
                  if (filein.fail()) {
                    errorCULQCD("ERROR: Unable to read file: %s in line %d\n",
                                filename.c_str(), line);
                  }
                }
              array.Set(m, idx);
            }
          }
    COUT << "Finisning reading configuration..." << endl;
  } else {
    errorCULQCD("Error reading configuration: %s\n", filename.c_str());
  }
  filein.close();
  if (arrayin.Mode() == Device) {
    CUDA_SAFE_CALL(
        cudaMemcpy(arrayin.GetPtr(), array.GetPtr(),
                   sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                   cudaMemcpyHostToDevice));
    array.Release();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void SaveBin_Gauge<float, float>(gauges arrayin, string filename,
                                          bool withheader);
template void SaveBin_Gauge<float, double>(gauges arrayin, string filename,
                                           bool withheader);
template void SaveBin_Gauge<double, float>(gauged arrayin, string filename,
                                           bool withheader);
template void SaveBin_Gauge<double, double>(gauged arrayin, string filename,
                                            bool withheader);

template <class Real, class RealSaveConf>
void SaveBin_Gauge(gauge arrayin, string filename, bool withheader) {
// if(!arrayin.EvenOdd()) errorCULQCD("Not implemented saving a non even/odd
// array format...\n");
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    if (!arrayin.Border())
      errorCULQCD("Not implemented saving a non gauge array format without "
                  "border in multi-GPU mode...\n");
    gauge array = arrayin;
    if (arrayin.Mode() == Device) {
      gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
                arrayin.Border());
      CUDA_SAFE_CALL(
          cudaMemcpy(A_h.GetPtr(), arrayin.GetPtr(),
                     sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                     cudaMemcpyDeviceToHost));
      array = A_h;
    }
    ofstream fileout;
    if (mynode() == masternode()) {
      fileout.open(filename.c_str(), ios::binary | ios::out);
      // fileout.precision(14);
      // fileout.setf(ios_base::scientific);
      if (!fileout.is_open()) {
        errorCULQCD("Error saving configuration: %s\n", filename.c_str());
      }
    }
    COUT << "Saving configuration " << filename << endl;
    msun *data = (msun *)safe_malloc(sizeof(msun) * 4);

    _matrixsun<RealSaveConf, NCOLORS> *tmp;
    if (mynode() == masternode() && (sizeof(Real)) != (sizeof(RealSaveConf)))
      tmp = (_matrixsun<RealSaveConf, NCOLORS> *)safe_malloc(
          sizeof(_matrixsun<RealSaveConf, NCOLORS>) * 4);

    if (mynode() == masternode() && withheader) {
      fileout.write((const char *)(&PARAMS::Grid), sizeof(int) * 4);
      fileout.write((const char *)(&PARAMS::Beta), sizeof(Real));
      size_t confprec = sizeof(RealSaveConf);
      fileout.write((const char *)(&confprec), sizeof(size_t));
    }
    for (int t = 0; t < PARAMS::NT; t++)
      for (int k = 0; k < PARAMS::NZ; k++)
        for (int j = 0; j < PARAMS::NY; j++)
          for (int i = 0; i < PARAMS::NX; i++) {
            int x[4];
            x[0] = i % param_Grid(0);
            x[1] = j % param_Grid(1);
            x[2] = k % param_Grid(2);
            x[3] = t % param_Grid(3);
            for (int id = 0; id < 4; id++) {
              x[id] += param_border(id);
            }
            int parity = (i + j + k + t) & 1;
            int id = x[0] + x[1] * param_GridG(0) +
                     x[2] * param_GridG(0) * param_GridG(1);
            id += x[3] * param_GridG(0) * param_GridG(1) * param_GridG(2);
            if (arrayin.EvenOdd()) {
              id = id >> 1;
              id += parity * param_HalfVolumeG();
            }

            int nodetorecv = node_number(i, j, k, t);
            if (mynode() == nodetorecv) {
              for (int dir = 0; dir < 4; dir++) {
                int idx = id + dir * param_VolumeG();
                data[dir] = array.Get(idx);
              }
            }
            if (mynode() == nodetorecv && mynode() != masternode()) {
              MPI_Send(data, NCOLORS * NCOLORS * 4, mpi_datatype<complex>(),
                       masternode(), mynode(), MPI_COMM_WORLD);
            }
            if (mynode() != nodetorecv && mynode() == masternode()) {
              MPI_Recv(data, NCOLORS * NCOLORS * 4, mpi_datatype<complex>(),
                       nodetorecv, nodetorecv, MPI_COMM_WORLD, &MPI_StatuS);
            }
            if (mynode() == masternode()) {
              if (sizeof(Real) != (sizeof(RealSaveConf))) {
                RealSaveConf *p = reinterpret_cast<RealSaveConf *>(tmp);
                for (int it = 0; it < 8 * NCOLORS * NCOLORS; it++)
                  p[it] = (RealSaveConf)(reinterpret_cast<Real *>(data))[it];
                fileout.write((const char *)tmp,
                              sizeof(_matrixsun<RealSaveConf, NCOLORS>) * 4);
              } else
                fileout.write((const char *)data,
                              sizeof(_matrixsun<Real, NCOLORS>) * 4);
              if (fileout.fail()) {
                errorCULQCD("ERROR: Unable save to file: %s\n",
                            filename.c_str());
              }
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
    host_free(data);
    if (mynode() == masternode() && (sizeof(Real)) != (sizeof(RealSaveConf)))
      host_free(tmp);
    COUT << "Finisning saving configuration..." << endl;
    if (mynode() == masternode())
      fileout.close();
    if (arrayin.Mode() == Device) {
      array.Release();
    }
    return;
  }
#endif
  gauge array = arrayin;
  if (arrayin.Mode() == Device) {
    gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
              arrayin.Border());
    CUDA_SAFE_CALL(
        cudaMemcpy(A_h.GetPtr(), arrayin.GetPtr(),
                   sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                   cudaMemcpyDeviceToHost));
    array = A_h;
  }
  ofstream fileout;
  fileout.open(filename.c_str(), ios::binary | ios::out);
  if (fileout.is_open()) {
    cout << "Saving configuration " << filename << endl;
    msun *data = (msun *)safe_malloc(sizeof(msun) * NDIMS);
    _matrixsun<RealSaveConf, NCOLORS> *tmp;
    if (sizeof(Real) != (sizeof(RealSaveConf)))
      tmp = (_matrixsun<RealSaveConf, NCOLORS> *)safe_malloc(
          sizeof(_matrixsun<RealSaveConf, NCOLORS>) * NDIMS);

    if (withheader) {
      fileout.write((const char *)(&PARAMS::Grid), sizeof(int) * NDIMS);
      fileout.write((const char *)(&PARAMS::Beta), sizeof(Real));
      size_t confprec = sizeof(RealSaveConf);
      fileout.write((const char *)(&confprec), sizeof(size_t));
    }

    int x[NDIMS];
    for (int i = 0; i < param_Volume(); i++) {
      Index_ND_NM(i, x);

      int x_sum = 0;
      for (int i = 0; i < NDIMS; i++)
        x_sum += x[i];
      int parity = x_sum & 1;

      int id = i;
      if (arrayin.EvenOdd()) {
        id = id >> 1;
        id += parity * param_HalfVolume();
      }

      for (int dir = 0; dir < NDIMS; dir++) {
        int idx = id + dir * param_Volume();
        data[dir] = array.Get(idx);
      }
      if (sizeof(Real) != (sizeof(RealSaveConf))) {
        RealSaveConf *p = reinterpret_cast<RealSaveConf *>(tmp);
        for (int it = 0; it < 8 * NCOLORS * NCOLORS; it++)
          p[it] = (RealSaveConf)(reinterpret_cast<Real *>(data))[it];
        fileout.write((const char *)tmp,
                      sizeof(_matrixsun<RealSaveConf, NCOLORS>) * NDIMS);
      } else
        fileout.write((const char *)data,
                      sizeof(_matrixsun<Real, NCOLORS>) * NDIMS);
      if (fileout.fail()) {
        errorCULQCD("ERROR: Unable save to file: %s\n", filename.c_str());
      }
    }

    host_free(data);
    if (sizeof(Real) != (sizeof(RealSaveConf)))
      host_free(tmp);
    COUT << "Finisning saving configuration..." << endl;
  } else {
    errorCULQCD("Error saving configuration: %s\n", filename.c_str());
  }
  fileout.close();
  if (arrayin.Mode() == Device)
    array.Release();
}

template <class Real> bool checkParams(int gridDim[NDIMS], Real beta) {
  COUT << "Parameters: " << endl;
  for (int dim = 0; dim < NDIMS; dim++)
    COUT << "\tdim " << dim << ": " << gridDim[dim]
         << "\t set: " << PARAMS::Grid[dim] << endl;
  COUT << "\tBeta: " << beta << "\t set: " << (Real)PARAMS::Beta << endl;
  for (int dim = 0; dim < NDIMS; dim++)
    if (gridDim[dim] != PARAMS::Grid[dim])
      return true;
  if ((Real)PARAMS::Beta != beta)
    return true;
  return false;
}

template void ReadBin_Gauge<float, float>(gauges arrayin, std::string filename,
                                          bool withheader);
template void ReadBin_Gauge<float, double>(gauges arrayin, std::string filename,
                                           bool withheader);
template void ReadBin_Gauge<double, float>(gauged arrayin, std::string filename,
                                           bool withheader);
template void ReadBin_Gauge<double, double>(gauged arrayin,
                                            std::string filename,
                                            bool withheader);

template <class Real, class RealSaveConf>
void ReadBin_Gauge(gauge arrayin, std::string filename, bool withheader) {
// if(!arrayin.EvenOdd()) errorCULQCD("Not implemented reading a non even/odd
// array format...\n");
#ifdef MULTI_GPU
  if (numnodes() > 1) {
    if (!arrayin.Border())
      errorCULQCD("Not implemented reading a non gauge array format without "
                  "border in multi-GPU mode...\n");
    gauge array = arrayin;
    if (arrayin.Mode() == Device) {
      gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
                arrayin.Border());
      array = A_h;
    }
    ifstream filein;
    if (mynode() == masternode()) {
      filein.open(filename.c_str(), ios::binary | ios::in);
      if (!filein.is_open()) {
        errorCULQCD("Error reading configuration: %s\n", filename.c_str());
      }
    }
    int line = 0;
    COUT << "Reading configuration " << filename << endl;
    msun *data = (msun *)safe_malloc(sizeof(msun) * 4);
    _matrixsun<RealSaveConf, NCOLORS> *tmp;
    if (mynode() == masternode() && (sizeof(Real)) != (sizeof(RealSaveConf)))
      tmp = (_matrixsun<RealSaveConf, NCOLORS> *)safe_malloc(
          sizeof(_matrixsun<RealSaveConf, NCOLORS>) * 4);

    if (mynode() == masternode() && withheader) {
      int gridDim[4];
      filein.read((char *)(&gridDim), sizeof(int) * 4);
      for (int dim = 0; dim < 4; dim++)
        COUT << "dim " << dim << " : " << gridDim[dim] << endl;
      Real beta = 0.0;
      filein.read((char *)(&beta), sizeof(Real));
      COUT << "beta: " << beta << endl;
      size_t confprec;
      filein.read((char *)(&confprec), sizeof(size_t));
      COUT << "prec: " << confprec << endl;
      if (confprec != sizeof(RealSaveConf))
        errorCULQCD("Error: Input lattice precision does not match: %s with "
                    "precision %d\n",
                    filename.c_str(), confprec);

      bool val = checkParams<Real>(gridDim, (Real)beta);
      if (val)
        errorCULQCD("Error: Input lattice parameters does not match with read "
                    "configuration: %s\n",
                    filename.c_str());
    }
    for (int t = 0; t < PARAMS::NT; t++)
      for (int k = 0; k < PARAMS::NZ; k++)
        for (int j = 0; j < PARAMS::NY; j++)
          for (int i = 0; i < PARAMS::NX; i++) {
            int x[4];
            x[0] = i % param_Grid(0);
            x[1] = j % param_Grid(1);
            x[2] = k % param_Grid(2);
            x[3] = t % param_Grid(3);
            for (int id = 0; id < 4; id++) {
              x[id] += param_border(id);
            }
            int parity = (i + j + k + t) & 1;
            int id = x[0] + x[1] * param_GridG(0) +
                     x[2] * param_GridG(0) * param_GridG(1);
            id += x[3] * param_GridG(0) * param_GridG(1) * param_GridG(2);
            if (arrayin.EvenOdd()) {
              id = id >> 1;
              id += parity * param_HalfVolumeG();
            }

            if (mynode() == masternode()) {
              if (sizeof(Real) != (sizeof(RealSaveConf))) {
                filein.read((char *)tmp,
                            sizeof(_matrixsun<RealSaveConf, NCOLORS>) * 4);
                for (int it = 0; it < 8 * NCOLORS * NCOLORS; it++)
                  (reinterpret_cast<Real *>(data))[it] =
                      (Real) reinterpret_cast<RealSaveConf *>(tmp)[it];

              } else
                filein.read((char *)data, sizeof(msun) * 4);
              if (filein.fail()) {
                errorCULQCD("ERROR: Unable read file: %s\n", filename.c_str());
              }
            }
            for (int dir = 0; dir < 4; dir++) {
              int idx = id + dir * param_VolumeG();
              msun m;
              if (mynode() == masternode()) {
                m = data[dir];
              }
              int nodetosend = node_number(i, j, k, t);
              if (mynode() == nodetosend && mynode() == masternode()) {
                array.Set(m, idx);
              }
              if (mynode() != nodetosend && mynode() == masternode()) {
                MPI_Send(&m, NCOLORS * NCOLORS, mpi_datatype<complex>(),
                         nodetosend, nodetosend, MPI_COMM_WORLD);
              }
              if (mynode() == nodetosend && mynode() != masternode()) {
                msun mm;
                MPI_Recv(&mm, NCOLORS * NCOLORS, mpi_datatype<complex>(),
                         masternode(), mynode(), MPI_COMM_WORLD, &MPI_StatuS);
                array.Set(mm, idx);
              }
            }
          }
    host_free(data);
    if (mynode() == masternode() && ((sizeof(Real)) != (sizeof(RealSaveConf))))
      host_free(tmp);
    COUT << "Finisning reading configuration..." << endl;
    if (mynode() == masternode())
      filein.close();
    if (arrayin.Mode() == Device) {
      CUDA_SAFE_CALL(cudaMemcpy(arrayin.GetPtr(), array.GetPtr(),
                                arrayin.Bytes(), cudaMemcpyHostToDevice));
      array.Release();
    }
    CUDA_SAFE_DEVICE_SYNC();
    for (int parity = 0; parity < 2; ++parity)
      for (int mu = 0; mu < 4; ++mu) {
        Exchange_gauge_border_links_gauge(arrayin, mu, parity);
      }
    return;
  }
#endif
  gauge array = arrayin;
  if (arrayin.Mode() == Device) {
    gauge A_h(arrayin.Type(), Host, arrayin.Size(), arrayin.EvenOdd(),
              arrayin.Border());
    array = A_h;
  }
  ifstream filein;
  filein.open(filename.c_str(), ios::binary | ios::in);
  if (filein.is_open()) {
    cout << "Reading configuration " << filename << endl;
    msun *data = (msun *)safe_malloc(sizeof(msun) * NDIMS);
    _matrixsun<RealSaveConf, NCOLORS> *tmp;
    if (sizeof(Real) != (sizeof(RealSaveConf)))
      tmp = (_matrixsun<RealSaveConf, NCOLORS> *)safe_malloc(
          sizeof(_matrixsun<RealSaveConf, NCOLORS>) * NDIMS);

    if (withheader) {
      int gridDim[NDIMS];
      filein.read((char *)(&gridDim), sizeof(int) * NDIMS);
      for (int dim = 0; dim < NDIMS; dim++)
        COUT << "dim " << dim << " : " << gridDim[dim] << endl;
      Real beta = 0.0;
      filein.read((char *)(&beta), sizeof(Real));
      COUT << "beta: " << beta << endl;
      size_t confprec;
      filein.read((char *)(&confprec), sizeof(size_t));
      COUT << "prec: " << confprec << endl;
      if (confprec != sizeof(RealSaveConf))
        errorCULQCD("Error: Input lattice precision does not match: %s with "
                    "precision %d\n",
                    filename.c_str(), confprec);

      bool val = checkParams<Real>(gridDim, (Real)beta);
      if (val)
        errorCULQCD("Error: Input lattice parameters does not match with read "
                    "configuration: %s\n",
                    filename.c_str());
    }
    int x[NDIMS];
    for (int i = 0; i < param_Volume(); i++) {
      Index_ND_NM(i, x);

      int x_sum = 0;
      for (int i = 0; i < NDIMS; i++)
        x_sum += x[i];
      int parity = x_sum & 1;

      int id = i;
      if (arrayin.EvenOdd()) {
        id = id >> 1;
        id += parity * param_HalfVolume();
      }
      if (sizeof(Real) != sizeof(RealSaveConf)) {
        filein.read((char *)tmp,
                    sizeof(_matrixsun<RealSaveConf, NCOLORS>) * NDIMS);
        for (int it = 0; it < 8 * NCOLORS * NCOLORS; it++)
          (reinterpret_cast<Real *>(data))[it] =
              (Real) reinterpret_cast<RealSaveConf *>(tmp)[it];

      } else
        filein.read((char *)data, sizeof(msun) * NDIMS);
      if (filein.fail()) {
        errorCULQCD("ERROR: Unable read file: %s\n", filename.c_str());
      }
      for (int dir = 0; dir < NDIMS; dir++) {
        int idx = id + dir * param_Volume();
        array.Set(data[dir], idx);
      }
    }

    host_free(data);
    if (sizeof(Real) != (sizeof(RealSaveConf)))
      host_free(tmp);
    COUT << "Finisning reading configuration..." << endl;
  } else {
    errorCULQCD("Error reading configuration: %s\n", filename.c_str());
  }
  filein.close();
  if (arrayin.Mode() == Device) {
    CUDA_SAFE_CALL(
        cudaMemcpy(arrayin.GetPtr(), array.GetPtr(),
                   sizeof(complex) * arrayin.Size() * arrayin.getNumElems(),
                   cudaMemcpyHostToDevice));
    array.Release();
  }
}

} // namespace CULQCD
