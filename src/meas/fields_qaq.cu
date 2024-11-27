#if (NDIMS == 4)
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <comm_mpi.h>
#include <constants.h>
#include <cuda_common.h>
#include <device_PHB_OVR.h>
#include <device_load_save.h>
#include <exchange.h>
#include <gaugearray.h>
#include <index.h>
#include <matrixsun.h>
#include <monte.h>
#include <reunitlink.h>
#include <staple.h>
#include <texture_host.h>
#include <timer.h>

#include <sharedmemtypes.h>

#include <launch_kernel.cuh>
#include <tune.h>

#include <cudaAtomic.h>

#include <cub/cub.cuh>

using namespace std;

namespace CULQCD {

// From normal to normal lattice index
inline __host__ __device__ void get4DFL(int id, int x[4]) {
  x[3] = id / (param_Grid(0) * param_Grid(1) * param_Grid(2));
  x[2] = (id / (param_Grid(0) * param_Grid(1))) % param_Grid(2);
  x[1] = (id / param_Grid(0)) % param_Grid(1);
  x[0] = id % param_Grid(0);
}
inline __host__ __device__ int neighborIndexFL(int id, int mu, int lmu, int nu,
                                               int lnu) {
  int x[4];
  get4DFL(id, x);
  x[mu] = periodic(x[mu] + lmu, param_Grid(mu));
  x[nu] = periodic(x[nu] + lnu, param_Grid(nu));
  return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1]) *
             param_Grid(0) +
         x[0];
}
inline __host__ __device__ int neighborIndexFL(int id, int mu, int lmu) {
  int x[4];
  get4DFL(id, x);
  x[mu] = periodic(x[mu] + lmu, param_Grid(mu));
  return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1]) *
             param_Grid(0) +
         x[0];
}

__device__ __host__ inline int linkIndex2(int x[], int dx[]) {
  int y[4];
  for (int i = 0; i < 4; i++)
    y[i] = (x[i] + dx[i] + param_Grid(i)) % param_Grid(i);
  return (((y[3] * param_Grid(2) + y[2]) * param_Grid(1) + y[1]) *
              param_Grid(0) +
          y[0]);
}

inline __host__ __device__ int neighborIndex4DFL(int id, int mu, int lmu) {
  int x[4];
  get4DFL(id, x);
  x[mu] = (x[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  return (((x[3] * param_Grid(2) + x[2]) * param_Grid(1)) + x[1]) *
             param_Grid(0) +
         x[0];
}
inline __host__ __device__ int neighborIndex4DFL(int x[3], int mu, int lmu,
                                                 int nu, int lnu) {
  int y[4];
  for (int dir = 0; dir < 4; dir++)
    y[dir] = x[dir];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  y[nu] = (y[nu] + lnu + param_Grid(nu)) % param_Grid(nu);
  return (((y[3] * param_Grid(2) + y[2]) * param_Grid(1) + y[1]) *
              param_Grid(0) +
          y[0]);
}

inline __host__ __device__ int neighborIndex4DFL(int x[3], int mu, int lmu) {
  int y[4];
  for (int dir = 0; dir < 4; dir++)
    y[dir] = x[dir];
  y[mu] = (y[mu] + lmu + param_Grid(mu)) % param_Grid(mu);
  return (((y[3] * param_Grid(2) + y[2]) * param_Grid(1) + y[1]) *
              param_Grid(0) +
          y[0]);
}

template <class Real> struct FieldWLArg {
  const complex *gaugefield;
  const complex *wilson_spaceline;
  Real *plaq;
  Real *wloop;
  Real *field;
  int radius;
  int Tmax;
  int nx;
  int ny;
};

#if __CUDA_ARCH__ >= 350
// Device has ldg
template <typename T> __device__ __forceinline__ T Aldg(const T *ptr) {
  return __ldg(ptr);
}

#else
// Device does not, fall back.
template <typename T> __device__ __forceinline__ T Aldg(const T *ptr) {
  return *ptr;
}
#endif

template <bool UseTex, class Real, bool planexy, typename BlockReduce,
          typename TempStorage>
__device__ inline void CalcField(FieldWLArg<Real> arg, int id,
                                 TempStorage &temp_storage, Real w,
                                 int fieldoffset, int it, int mu) {

  int dir2 = mu;
  int xx[4];
  get4DFL(id, xx);
  xx[dir2] = (xx[dir2] + arg.radius / 2) % DEVPARAMS::Grid[dir2];
  xx[3] = (xx[3] + it / 2) % DEVPARAMS::Grid[3];

  if (planexy) {
    for (int ix = 0; ix < arg.nx; ++ix)
      for (int iy = 0; iy < arg.ny; ++iy) {

        Real field[6];
        for (int dd = 0; dd < 6; dd++)
          field[dd] = 0.0;

        for (int dir1 = 0; dir1 < 3; dir1++) {
          if (dir1 == dir2)
            continue;
          int dir3 = 0;
          for (int dir33 = 0; dir33 < 3; dir33++)
            if (dir1 != dir33 && dir2 != dir33)
              dir3 = dir33;

          int s = neighborIndex4DFL(xx, dir1, ix - arg.nx / 2, dir2,
                                    iy - arg.ny / 2);

          if (id < DEVPARAMS::Volume) {
            // Ex^2
            // Real plaq = arg.plaq[s + dir1 * DEVPARAMS::Volume];
            Real plaq = Aldg(arg.plaq + s + dir1 * DEVPARAMS::Volume);
            field[0] += plaq;
            // Ey^2
            // plaq = arg.plaq[s + dir2 * DEVPARAMS::Volume];
            plaq = Aldg(arg.plaq + s + dir2 * DEVPARAMS::Volume);
            field[1] += plaq;
            // Ez^2
            int s1 = neighborIndex4DFL(s, dir3, -1);
            // plaq = arg.plaq[s + dir3 * DEVPARAMS::Volume];
            // plaq += arg.plaq[s1 + dir3 * DEVPARAMS::Volume];
            plaq = Aldg(arg.plaq + s + dir3 * DEVPARAMS::Volume);
            plaq += Aldg(arg.plaq + s1 + dir3 * DEVPARAMS::Volume);
            field[2] += plaq * 0.5;
            // Bx^2
            // plaq = arg.plaq[s + (3 + dir1) * DEVPARAMS::Volume];
            // plaq += arg.plaq[s1 + (3 + dir1) * DEVPARAMS::Volume];
            plaq = Aldg(arg.plaq + s + (3 + dir1) * DEVPARAMS::Volume);
            plaq += Aldg(arg.plaq + s1 + (3 + dir1) * DEVPARAMS::Volume);
            field[3] += plaq * 0.5;
            // By^2
            // plaq = arg.plaq[s + (3 + dir2) * DEVPARAMS::Volume];
            // plaq += arg.plaq[s1 + (3 + dir2) * DEVPARAMS::Volume];
            plaq = Aldg(arg.plaq + s + (3 + dir2) * DEVPARAMS::Volume);
            plaq += Aldg(arg.plaq + s1 + (3 + dir2) * DEVPARAMS::Volume);
            field[4] += plaq * 0.5;
            // Bz^2
            // plaq = arg.plaq[s + (3 + dir3) * DEVPARAMS::Volume];
            plaq = Aldg(arg.plaq + s + (3 + dir3) * DEVPARAMS::Volume);
            field[5] += plaq;
          }
        }
        int fieldstride = it * arg.nx * arg.ny;
        for (int dd = 0; dd < 6; dd++)
          field[dd] *= w;
        Real aggregate[6];
        for (int dd = 0; dd < 6; dd++) {
          aggregate[dd] =
              BlockReduce(temp_storage).Reduce(field[dd], Summ<Real>());
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          // accum Ex^2
          int id0 = ix + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id0, aggregate[0]);
          int id1 = ((ix + 1) % arg.nx) + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1, aggregate[0]);
          // accum Ey^2
          CudaAtomicAdd(arg.field + id0 + fieldoffset, aggregate[1]);
          id1 = ix + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + fieldoffset, aggregate[1]);
          // accum Ez^2
          CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset, aggregate[2]);
          // accum Bx^2
          CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset, aggregate[3]);
          id1 = ix + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset, aggregate[3]);
          // accum By^2
          CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset, aggregate[4]);
          id1 = ((ix + 1) % arg.nx) + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
          // accum Bz^2
          CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset, aggregate[5]);
          id1 = ((ix + 1) % arg.nx) + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);

          id1 = ix + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
          id1 =
              ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
        }
      }
  } else {
    for (int ix = 0; ix < arg.nx; ++ix)
      for (int iy = 0; iy < arg.ny; ++iy) {

        Real field[6];
        for (int dd = 0; dd < 6; dd++)
          field[dd] = 0.0;

        for (int dir1 = 0; dir1 < 3; dir1++) {
          if (dir1 == dir2)
            continue;
          int dir3 = 0;
          for (int dir33 = 0; dir33 < 3; dir33++)
            if (dir1 != dir33 && dir2 != dir33)
              dir3 = dir33;

          int s = neighborIndex4DFL(xx, dir1, ix - arg.nx / 2, dir3,
                                    iy - arg.ny / 2);

          if (id < DEVPARAMS::Volume) {
            // Ex^2
            Real plaq = Aldg(arg.plaq + s + dir1 * DEVPARAMS::Volume);
            field[0] += plaq;
            // Ey^2
            int s1 = neighborIndex4DFL(s, dir2, -1);
            plaq = Aldg(arg.plaq + s + dir2 * DEVPARAMS::Volume);
            plaq += Aldg(arg.plaq + s1 + dir2 * DEVPARAMS::Volume);
            field[1] += plaq * 0.5;
            // Ez^2
            plaq = Aldg(arg.plaq + s + dir3 * DEVPARAMS::Volume);
            field[2] += plaq;
            // Bx^2
            plaq = Aldg(arg.plaq + s + (3 + dir1) * DEVPARAMS::Volume);
            plaq += Aldg(arg.plaq + s1 + (3 + dir1) * DEVPARAMS::Volume);
            field[3] += plaq * 0.5;
            // By^2
            plaq = Aldg(arg.plaq + s + (3 + dir2) * DEVPARAMS::Volume);
            field[4] += plaq;
            // Bz^2
            plaq = Aldg(arg.plaq + s + (3 + dir3) * DEVPARAMS::Volume);
            plaq += Aldg(arg.plaq + s1 + (3 + dir3) * DEVPARAMS::Volume);
            field[5] += plaq * 0.5;
          }
        }
        int fieldstride = it * arg.nx * arg.ny;

        for (int dd = 0; dd < 6; dd++)
          field[dd] *= w;
        Real aggregate[6];
        for (int dd = 0; dd < 6; dd++) {
          aggregate[dd] =
              BlockReduce(temp_storage).Reduce(field[dd], Summ<Real>());
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          // accum Ex^2
          int id0 = ix + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id0, aggregate[0]);
          int id1 = ((ix + 1) % arg.nx) + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1, aggregate[0]);
          // accum Ey^2
          CudaAtomicAdd(arg.field + id0 + fieldoffset, aggregate[1]);
          // accum Ez^2
          CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset, aggregate[2]);
          id1 = ix + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 2 * fieldoffset, aggregate[2]);
          // accum Bx^2
          CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset, aggregate[3]);
          id1 = ix + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset, aggregate[3]);
          // accum By^2
          CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset, aggregate[4]);
          id1 = ((ix + 1) % arg.nx) + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
          id1 = ix + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
          id1 =
              ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
          // accum Bz^2
          CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset, aggregate[5]);
          id1 = ((ix + 1) % arg.nx) + iy * arg.nx + fieldstride;
          CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
        }
      }
  }
}

template <int blockSize, bool UseTex, class Real, ArrayType atype, bool planexy>
__global__ void kernel_FieldWilsonLoop(FieldWLArg<Real> arg) {

  typedef cub::BlockReduce<Real, blockSize> BlockReduce;
  typedef typename BlockReduce::TempStorage BlockReduceTempStorage;
  __shared__ BlockReduceTempStorage temp_storage;

  int id = INDEX1D();

  // int muvolume = arg.mu * DEVPARAMS::Volume;
  int tdirvolume = 3 * DEVPARAMS::Volume;
  int gfoffset = 3 * DEVPARAMS::Volume;
  int fieldoffset = (arg.Tmax + 1) * arg.nx * arg.ny;

  for (int mu = 0; mu < 3; mu++) {
    int x[4];
    get4DFL(id, x);

    int idl = (x[2] * param_Grid(1) + x[1]) * param_Grid(0) +
              x[0]; // space index left
    x[mu] = periodic(x[mu] + arg.radius, param_Grid(mu));
    int idr = (x[2] * param_Grid(1) + x[1]) * param_Grid(0) +
              x[0]; // space index right

    msun t0 = msun::identity();
    msun t1 = msun::identity();
    for (int it = 0; it <= arg.Tmax; it++) {
      int idt = periodic(x[3] + it, param_Grid(3));
      idt *= DEVPARAMS::tstride;

      Real w = 0.0;
      msun linkb = msun::zero();
      if (id < DEVPARAMS::Volume)
        linkb = GAUGE_LOAD<false, atype, Real>(arg.wilson_spaceline,
                                               id + mu * DEVPARAMS::Volume,
                                               gfoffset); // bottom space links

      if (id < DEVPARAMS::Volume) {
        msun linkt = GAUGE_LOAD_DAGGER<false, atype, Real>(
            arg.wilson_spaceline, idl + idt + mu * DEVPARAMS::Volume,
            gfoffset); // top space links
        w = (linkb * t1 * linkt * t0.dagger()).realtrace();
      }

      if (id < DEVPARAMS::Volume && it < arg.Tmax) {
        t0 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, idl + idt + tdirvolume, DEVPARAMS::size);
        t1 *= GAUGE_LOAD<UseTex, atype, Real>(
            arg.gaugefield, idr + idt + tdirvolume, DEVPARAMS::size);
      }
      // fields!!!
      CalcField<UseTex, Real, planexy, BlockReduce, BlockReduceTempStorage>(
          arg, id, temp_storage, w, fieldoffset, it, mu);
      // save wilson loop
      Real aggregate = BlockReduce(temp_storage).Reduce(w, Summ<Real>());
      if (threadIdx.x == 0)
        CudaAtomicAdd(arg.wloop + it, aggregate);
      __syncthreads();
    }
  }
}

template <bool UseTex, class Real, ArrayType atype, bool planexy>
class FieldWilsonLoop : Tunable {
private:
  FieldWLArg<Real> arg;
  gauge array;
  int size;
  double timesec;
#ifdef TIMMINGS
  Timer FieldWilsonLooptime;
#endif
  TuneParam tp;

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream) {
    tp = tuneLaunch(*this, getTuning(), getVerbosity());

    size_t wloop_mem = (arg.Tmax + 1) * sizeof(Real);
    size_t field_mem = arg.nx * arg.ny * 6 * (arg.Tmax + 1) * sizeof(Real);
    CUDA_SAFE_CALL(cudaMemset(arg.wloop, 0, wloop_mem));
    CUDA_SAFE_CALL(cudaMemset(arg.field, 0, field_mem));

    LAUNCH_KERNEL(kernel_FieldWilsonLoop, tp, stream, arg, UseTex, Real, atype,
                  planexy);
  }

public:
  FieldWilsonLoop(FieldWLArg<Real> arg, gauge array) : arg(arg), array(array) {
    size = 1;
    for (int i = 0; i < 4; i++) {
      size *= PARAMS::Grid[i];
    }
    timesec = 0.0;
  }
  void SetIop(int iop) { arg.iop = iop; }
  ~FieldWilsonLoop() {};
  void Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
    FieldWilsonLooptime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
#ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC();
    FieldWilsonLooptime.stop();
    timesec = FieldWilsonLooptime.getElapsedTimeInSec();
#endif
  }
  void Run() { return Run(0); }
  double flops() { return ((double)flop() * 1.0e-9) / timesec; }
  double bandwidth() { return (double)bytes() / (timesec * (double)(1 << 30)); }
  long long flop() const { return 0; }
  long long bytes() const { return 0; }
  double time() { return timesec; }
  void stat() {
    COUT << "FieldWilsonLoop:  " << time() << " s\t" << bandwidth() << " GB/s\t"
         << flops() << " GFlops" << endl;
  }
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec=" << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(),
                   aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << ","
       << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {}
  void postTune() {}
};

template <bool UseTex, class Real, bool planexy>
void CalcFieldWilsonLoop_dg(gauge array, gauge wilson_spaceline,
                            Real *plaqfield, Real *wloop, Real *field,
                            int radius, int Tmax, int nx, int ny) {
  Timer mtime;
  mtime.start();
  FieldWLArg<Real> arg;
  arg.gaugefield = array.GetPtr();
  arg.wilson_spaceline = wilson_spaceline.GetPtr();
  arg.plaq = plaqfield;
  arg.wloop = wloop;
  arg.field = field;
  arg.radius = radius;
  arg.Tmax = Tmax;
  arg.nx = nx;
  arg.ny = ny;

  if (array.Type() != SOA || wilson_spaceline.Type() != SOA)
    errorCULQCD("Only defined for SOA arrays...\n");
  if (array.EvenOdd() == true || wilson_spaceline.EvenOdd() == true)
    errorCULQCD("Not defined for EvenOdd arrays...\n");

  FieldWilsonLoop<UseTex, Real, SOA, planexy> wl(arg, array);
  wl.Run();

  CUDA_SAFE_DEVICE_SYNC();
  mtime.stop();
  COUT << "Time FieldWilsonLoopF:  " << mtime.getElapsedTimeInSec() << " s"
       << endl;
}

template <bool UseTex, class Real>
void CalcFieldWilsonLoop_dg(gauge array, gauge wilson_spaceline,
                            Real *plaqfield, Real *wloop, Real *field,
                            int radius, int Tmax, int nx, int ny,
                            bool planexy) {
  if (planexy) {
    CalcFieldWilsonLoop_dg<UseTex, Real, true>(
        array, wilson_spaceline, plaqfield, wloop, field, radius, Tmax, nx, ny);
  } else
    CalcFieldWilsonLoop_dg<UseTex, Real, false>(
        array, wilson_spaceline, plaqfield, wloop, field, radius, Tmax, nx, ny);
}

template <class Real>
void CalcFieldWilsonLoop_dg(gauge array, gauge wilson_spaceline,
                            Real *plaqfield, Real *wloop, Real *field,
                            int radius, int Tmax, int nx, int ny,
                            bool planexy) {
  if (PARAMS::UseTex) {
    GAUGE_TEXTURE(array.GetPtr(), true);
    CalcFieldWilsonLoop_dg<true, Real>(array, wilson_spaceline, plaqfield,
                                       wloop, field, radius, Tmax, nx, ny,
                                       planexy);
  } else
    CalcFieldWilsonLoop_dg<false, Real>(array, wilson_spaceline, plaqfield,
                                        wloop, field, radius, Tmax, nx, ny,
                                        planexy);
}

template void CalcFieldWilsonLoop_dg<double>(
    gauged array, gauged wilson_spaceline, double *plaqfield, double *wloop,
    double *field, int radius, int Tmax, int nx, int ny, bool planexy);

} // namespace CULQCD
#endif
