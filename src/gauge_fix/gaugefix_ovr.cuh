
#ifndef GAUGEFIX_OVR_CUH
#define GAUGEFIX_OVR_CUH

#include "gaugefix_ovr_device.cuh"

#define GAUGE_FIXING_MAX_THREADS_FOR_8_THREADS_PER_SITE 768
#define GAUGE_FIXING_MAX_THREADS_FOR_4_THREADS_PER_SITE 512

#define LAUNCH_KERNEL_GFIX_8T(kernel, FuncType, tp, stream, arg, ...)          \
  switch (tp.block.x) {                                                        \
  case 256:                                                                    \
    kernel<FuncType, 32, __VA_ARGS__>                                          \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 512:                                                                    \
    kernel<FuncType, 64, __VA_ARGS__>                                          \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 768:                                                                    \
    kernel<FuncType, 96, __VA_ARGS__>                                          \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  default:                                                                     \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x);   \
  }
#define LAUNCH_KERNEL_GFIX_4T(kernel, FuncType, tp, stream, arg, ...)          \
  switch (tp.block.x) {                                                        \
  case 128:                                                                    \
    kernel<FuncType, 32, __VA_ARGS__>                                          \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 256:                                                                    \
    kernel<FuncType, 64, __VA_ARGS__>                                          \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 384:                                                                    \
    kernel<FuncType, 96, __VA_ARGS__>                                          \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 512:                                                                    \
    kernel<FuncType, 128, __VA_ARGS__>                                         \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 640:                                                                    \
    kernel<FuncType, 160, __VA_ARGS__>                                         \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  case 768:                                                                    \
    kernel<FuncType, 192, __VA_ARGS__>                                         \
        <<<tp.grid, tp.block.x, tp.shared_bytes, stream>>>(arg);               \
    break;                                                                     \
  default:                                                                     \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x);   \
  }
/*#define LAUNCH_KERNEL_GFIX_4T(kernel, FuncType, tp, stream, arg, ...)     \
  switch (tp.block.x) {             \
  case 128:                \
    kernel<FuncType, 32,__VA_ARGS__><<< tp.grid, tp.block.x, tp.shared_bytes,
  stream >>>(arg);   \
    break;                \
  case 256:                \
    kernel<FuncType, 64,__VA_ARGS__><<< tp.grid, tp.block.x, tp.shared_bytes,
  stream >>>(arg);   \
    break;                \
  case 384:                \
    kernel<FuncType, 96,__VA_ARGS__><<< tp.grid, tp.block.x, tp.shared_bytes,
  stream >>>(arg);   \
    break;                \
  case 512:               \
    kernel<FuncType, 128,__VA_ARGS__><<< tp.grid, tp.block.x, tp.shared_bytes,
  stream >>>(arg);   \
    break;              \
  default:                \
    errorCULQCD("%s not implemented for %d threads\n", #kernel, tp.block.x); \
    }*/
#define LAUNCH_KERNEL_GFIX(kernel, tp, stream, arg, ...)                       \
  switch (tp.block.z) {                                                        \
  case 0:                                                                      \
    LAUNCH_KERNEL_GFIX_8T(kernel, 0, tp, stream, arg, __VA_ARGS__);            \
    break;                                                                     \
  case 1:                                                                      \
    LAUNCH_KERNEL_GFIX_8T(kernel, 1, tp, stream, arg, __VA_ARGS__);            \
    break;                                                                     \
  case 2:                                                                      \
    LAUNCH_KERNEL_GFIX_8T(kernel, 2, tp, stream, arg, __VA_ARGS__);            \
    break;                                                                     \
  case 3:                                                                      \
    LAUNCH_KERNEL_GFIX_4T(kernel, 3, tp, stream, arg, __VA_ARGS__);            \
    break;                                                                     \
  case 4:                                                                      \
    LAUNCH_KERNEL_GFIX_4T(kernel, 4, tp, stream, arg, __VA_ARGS__);            \
    break;                                                                     \
  case 5:                                                                      \
    LAUNCH_KERNEL_GFIX_4T(kernel, 5, tp, stream, arg, __VA_ARGS__);            \
    break;                                                                     \
  default:                                                                     \
    errorCULQCD("%s not implemented for function type %d\n", #kernel,          \
                tp.block.z);                                                   \
  }

static dim3 Call_GaugeFix_createGrid(const dim3 &block, unsigned int size) {
  if (block.z <= 2)
    return GetBlockDim(block.x / 8, size);
  return GetBlockDim(block.x / 4, size);
}

static bool Call_GaugeFix_advanceBlockDim(TuneParam &param, unsigned int size,
                                          size_t precision) {
  // Use param.block.z to tune and save state for best kernel option
  //  to make use or not of CudaAtomicAdd operations and 4 or 8 threads per
  //  lattice site!!!
  const unsigned int atmadd = 0;
  unsigned int max_threads = GAUGE_FIXING_MAX_THREADS_FOR_8_THREADS_PER_SITE;
  unsigned int min_threads = 32 * 8;
  param.block.z += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT
                           // USING ATOMICADD, 4/8 threads per site
  if (param.block.z > 2) {
    min_threads = 32 * 4;
    max_threads = GAUGE_FIXING_MAX_THREADS_FOR_4_THREADS_PER_SITE;
  }
  param.block.x += min_threads;
  param.block.y = 1;
  param.grid = Call_GaugeFix_createGrid(param.block, size);
  if ((param.block.x >= min_threads) && (param.block.x <= max_threads)) {
    if (param.block.z == 0)
      param.shared_bytes = param.block.x * 4 * precision;
    else if (param.block.z == 1 || param.block.z == 2)
      param.shared_bytes = param.block.x * 4 * precision / 8;
    else if (param.block.z == 3)
      param.shared_bytes = param.block.x * 4 * precision;
    else if (param.block.z == 4 || param.block.z == 5)
      param.shared_bytes = param.block.x * precision;
    else
      errorCULQCD("Not implemented for option %d", param.block.z);
    return true;
  } else if (param.block.z == 0) {
    param.block.x = min_threads;
    param.block.y = 1;
    param.block.z = 1; // USE FOR ATOMIC ADD, 8 threads per lattice site
    param.grid = Call_GaugeFix_createGrid(param.block, size);
    param.shared_bytes = param.block.x * 4 * precision / 8;
    return true;
  } else if (param.block.z == 1) {
    param.block.x = min_threads;
    param.block.y = 1;
    param.block.z = 2; // USE FOR NO ATOMIC ADD and LESS SHARED MEM, 8 threads
                       // per lattice site
    param.grid = Call_GaugeFix_createGrid(param.block, size);
    param.shared_bytes = param.block.x * 4 * precision / 8;
    return true;
  } else if (param.block.z == 2) {
    param.block.x = min_threads;
    param.block.y = 1;
    param.block.z = 3; // USE FOR NO ATOMIC ADD, 4 threads per lattice site
    param.grid = Call_GaugeFix_createGrid(param.block, size);
    param.shared_bytes = param.block.x * 4 * precision;
    return true;
  } else if (param.block.z == 3) {
    param.block.x = min_threads;
    param.block.y = 1;
    param.block.z = 4; // USE FOR ATOMIC ADD, 4 threads per lattice site
    param.grid = Call_GaugeFix_createGrid(param.block, size);
    param.shared_bytes = param.block.x * precision;
    return true;
  } else if (param.block.z == 4) {
    param.block.x = min_threads;
    param.block.y = 1;
    param.block.z = 5; // USE FOR NO ATOMIC ADD and LESS SHARED MEM, 4 threads
                       // per lattice site
    param.grid = Call_GaugeFix_createGrid(param.block, size);
    param.shared_bytes = param.block.x * precision;
    return true;
  } else
    return false;
}

static unsigned int CalcSharedBytesPerBlock(const dim3 block, size_t prec) {
  if (block.z == 0)
    return block.x * 4 * prec;
  else if (block.z == 1 || block.z == 2)
    return block.x * 4 * prec / 8;
  else if (block.z == 3)
    return block.x * 4 * prec;
  else if (block.z == 4 || block.z == 5)
    return block.x * prec;
  else {
    errorCULQCD("Not implemented for option %d", block.z);
    return 0;
  }
}

static void CalcInitTuneParam(dim3 &block, dim3 &grid, int &shared_bytes,
                              unsigned int size, size_t prec) {
  block = dim3(256, 1, 0);
  grid = Call_GaugeFix_createGrid(block, size);
  shared_bytes = block.x * 4 * prec;
}

#ifdef GAUGE_FIXING_MAX_THREADS
#undef GAUGE_FIXING_MAX_THREADS
#endif

template <class Real> struct GaugeFixArg {
  complex *array;
  int size;
#ifdef MULTI_GPU
  int X[4];
  int border[4];
#endif
  Real relax_boost;
  int parity;
};

template <int FUNCTIONTYPE, int NTPVOL, int DIR, bool UseTex, ArrayType atype,
          class Real>
__global__ void kernel_do_hit_EO_shared_combo(GaugeFixArg<Real> arg) {
  // Get the local thread id for each site, since at each site we are assigning
  // 8 threads
  int tid = threadIdx.x % NTPVOL;
  // Get the global thread id
#if (__CUDA_ARCH__ >= 300)
  int ids = blockIdx.x * NTPVOL + tid;
#else
  int ids = gridDim.x * blockIdx.y + blockIdx.x;
  ids = NTPVOL * ids + tid;
#endif
  if (ids >= arg.size)
    return;
  // Get the direction, at this point the direction goes from 0-7
  // where 0-3 is for the uplinks and 4-7 to downlinks
  int mu = (threadIdx.x / NTPVOL);
  // 8 threads per lattice site
  if (FUNCTIONTYPE < 3) {
#ifdef MULTI_GPU
    int x[4];
    int oddbit = arg.parity;
    getEOCoords3(x, ids, DEVPARAMS::Grid, oddbit);
    for (int i = 0; i < 4; i++)
      x[i] += param_border(i);
    if (threadIdx.x >= NTPVOL * 4) { // Downlink index
      mu -= 4;
      x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
      oddbit = 1 - oddbit;
    }
    ids = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
               param_GridG(0) +
           x[0]) >>
          1;
    ids += oddbit * param_HalfVolumeG();
#else
    if (threadIdx.x >= NTPVOL * 4) { // Downlink index
      mu -= 4;
      ids = neighborEOIndexMinusOne(ids, arg.parity, mu);
    } else
      ids = EOIndeX(ids, arg.parity);
#endif
    ids += mu * DEVPARAMS::VolumeG;
    // Load uplink and downlink from global memory
    msun link =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, ids, DEVPARAMS::VolumeG * 4);
    // Gauge fix hits
    if (FUNCTIONTYPE == 0)
      GaugeFixHit_NoAtomicAdd<DIR, NTPVOL, Real>(link, arg.relax_boost, tid);
    if (FUNCTIONTYPE == 1)
      GaugeFixHit_AtomicAdd<DIR, NTPVOL, Real>(link, arg.relax_boost, tid);
    if (FUNCTIONTYPE == 2)
      GaugeFixHit_NoAtomicAdd_LessSM<DIR, NTPVOL, Real>(link, arg.relax_boost,
                                                        tid);
    GAUGE_SAVE<atype, Real>(arg.array, link, ids, DEVPARAMS::VolumeG * 4);
  }
  // 4 threads per lattice site
  else {
    int x[4];
    getEOCoords3(x, ids, DEVPARAMS::Grid, arg.parity);
#ifdef MULTI_GPU
    for (int i = 0; i < 4; i++)
      x[i] += param_border(i);
#endif
    ids = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
               param_GridG(0) +
           x[0]) >>
          1;
    ids += arg.parity * param_HalfVolumeG();
    ids += mu * DEVPARAMS::VolumeG;
    // Load uplink from global memory
    msun link =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, ids, DEVPARAMS::VolumeG * 4);
    x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
    int idl = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
                   param_GridG(0) +
               x[0]) >>
              1;
    idl += (1 - arg.parity) * param_HalfVolumeG();
    idl += mu * DEVPARAMS::VolumeG;
    // Load downlink from global memory
    msun link1 =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, idl, DEVPARAMS::VolumeG * 4);
    // Gauge fix hits
    if (FUNCTIONTYPE == 3)
      GaugeFixHit_NoAtomicAdd<DIR, NTPVOL, Real>(link, link1, arg.relax_boost,
                                                 tid);
    if (FUNCTIONTYPE == 4)
      GaugeFixHit_AtomicAdd<DIR, NTPVOL, Real>(link, link1, arg.relax_boost,
                                               tid);
    if (FUNCTIONTYPE == 5)
      GaugeFixHit_NoAtomicAdd_LessSM<DIR, NTPVOL, Real>(link, link1,
                                                        arg.relax_boost, tid);
    GAUGE_SAVE<atype, Real>(arg.array, link, ids, DEVPARAMS::VolumeG * 4);
    GAUGE_SAVE<atype, Real>(arg.array, link1, idl, DEVPARAMS::VolumeG * 4);
  }
}

template <int DIR, bool UseTex, ArrayType atype, class Real>
class GaugeFix_SingleNode : Tunable {
private:
  string functionName;
  gauge array;
  GaugeFixArg<Real> arg;
  double timesec;
  int grid[4];
#ifdef TIMMINGS
  Timer mtime;
#endif

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    LAUNCH_KERNEL_GFIX(kernel_do_hit_EO_shared_combo, tp, stream, arg, DIR,
                       UseTex, atype, Real);
  }
  dim3 createGrid(const dim3 &block) const {
    return Call_GaugeFix_createGrid(block, minThreads());
  }
  bool advanceBlockDim(TuneParam &param) const {
    return Call_GaugeFix_advanceBlockDim(param, minThreads(), sizeof(Real));
  }

private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const {
    return CalcSharedBytesPerBlock(param.block, sizeof(Real));
  }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.size; }

public:
  void Init() {
    arg.size = 1;
    for (int i = 0; i < 4; i++) {
      grid[i] = PARAMS::Grid[i];
      arg.size *= PARAMS::Grid[i];
    }
    arg.size = arg.size >> 1;
    timesec = 0.0;
    arg.parity = 0;
    arg.array = array.GetPtr();
  }
  GaugeFix_SingleNode(gauge &array, Real relax_boost) : array(array) {
    functionName = "OVR_GaugeFix";
    arg.relax_boost = relax_boost;
    Init();
  }

  ~GaugeFix_SingleNode() {};
  virtual void initTuneParam(TuneParam &param) const {
    CalcInitTuneParam(param.block, param.grid, param.shared_bytes, minThreads(),
                      sizeof(Real));
  }

  /** Sets default values for when tuning is disabled - this is guaranteed to
   * work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const {
    initTuneParam(param);
  }
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << grid[0] << "x";
    vol << grid[1] << "x";
    vol << grid[2] << "x";
    vol << grid[3];
    aux << "threads=" << arg.size << ",prec=" << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(),
                   array.ToStringArrayType().c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << ","
       << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { array.Backup(); }
  void postTune() { array.Restore(); }
  void Run(const cudaStream_t &stream) {
#ifdef TIMMINGS
    mtime.start();
#endif
    arg.parity = 0;
    apply(stream);
    arg.parity = 1;
    apply(stream);
#ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC();
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
  }
  void Run(const cudaStream_t &stream, int parityin) {
#ifdef TIMMINGS
    mtime.start();
#endif
    arg.parity = parityin;
    apply(stream);
#ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC();
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
  }
  void Run() { return Run(0); }
  double time() { return timesec; }
  void stat() {
    COUT << functionName << ":  " << time() << " s\t" << bandwidth()
         << " GB/s\t" << flops() << " GFlops" << endl;
  }

  long long flop() const {
    // using the flops from GaugeFixHit_NoAtomicAdd with 8 threads per lattice
    // site...
    return (TOTAL_SUB_BLOCKS * (22LL + 22LL * DIR + 224LL * NCOLORS) +
            8LL * array.getNumFlop(true) + 8LL * array.getNumFlop(false)) *
           arg.size * numnodes();
  }
  long long bytes() const {
    return 8LL * 2 * array.getNumParams() * arg.size * sizeof(Real) *
           numnodes();
  }

  double flops() { return ((double)flop() * 1.0e-9) / timesec; }

  double bandwidth() { return (double)bytes() / (timesec * (double)(1 << 30)); }
};
// #endif

#ifdef MULTI_GPU
template <int FUNCTIONTYPE, int NTPVOL, int DIR, bool UseTex, ArrayType atype,
          class Real>
__global__ void kernel_do_hit_EO_shared_combo_interior(GaugeFixArg<Real> arg) {
  // Get the local thread id for each site, since at each site we are assigning
  // 8 threads
  int tid = (threadIdx.x + NTPVOL) % NTPVOL;
  // Get the global thread id
#if (__CUDA_ARCH__ >= 300)
  int ids = blockIdx.x * NTPVOL + tid;
#else
  int ids = gridDim.x * blockIdx.y + blockIdx.x;
  ids = NTPVOL * ids + tid;
#endif
  if (ids >= arg.size)
    return;

  int x[4];
#ifdef MULTI_GPU
  int za = (ids / (arg.X[0] / 2));
  int zb = (za / arg.X[1]);
  x[1] = za - zb * arg.X[1];
  x[3] = (zb / arg.X[2]);
  x[2] = zb - x[3] * arg.X[2];
  int p = 0;
  for (int dr = 0; dr < 4; ++dr)
    p += arg.border[dr];
  p = p & 1;
  int x1odd = (x[1] + x[2] + x[3] + arg.parity + p) & 1;
  // int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
  x[0] = (2 * ids + x1odd) - za * arg.X[0];
  for (int dr = 0; dr < 4; ++dr)
    x[dr] += arg.border[dr];
#else
  getCoords3(x, ids, DEVPARAMS::Grid, arg.parity);
#endif
  int mu = (threadIdx.x / NTPVOL);

#ifdef ACTIVATE_COVARIANT_GAUGE_FIX_STR
  cuRNGState localState;
  int idr;
  if (threadIdx.x < NTPVOL) {
    idr = (((((x[3] - DEVPARAMS::Border[3]) * param_Grid(2) +
              (x[2] - DEVPARAMS::Border[2])) *
             param_Grid(1)) +
            x[1] - DEVPARAMS::Border[1]) *
               param_Grid(0) +
           x[0] - DEVPARAMS::Border[0]) >>
          1;
    localState = arg.state[idr];
  }
#endif

  if (FUNCTIONTYPE < 3) {
    int parity = arg.parity;
    if (threadIdx.x >= NTPVOL * 4) {
      mu -= 4;
      x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
      parity = 1 - parity;
    }
    ids = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
               param_GridG(0) +
           x[0]) >>
          1;
    ids += parity * DEVPARAMS::HalfVolumeG;
    ids += mu * DEVPARAMS::VolumeG;
    // Load uplink and downlink from global memory
    msun link =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, ids, DEVPARAMS::VolumeG * 4);
    // Gauge fix hits
    if (FUNCTIONTYPE == 0)
      GaugeFixHit_NoAtomicAdd<DIR, NTPVOL, Real>(link, arg.relax_boost, tid);
    if (FUNCTIONTYPE == 1)
      GaugeFixHit_AtomicAdd<DIR, NTPVOL, Real>(link, arg.relax_boost, tid);
    if (FUNCTIONTYPE == 2)
      GaugeFixHit_NoAtomicAdd_LessSM<DIR, NTPVOL, Real>(link, arg.relax_boost,
                                                        tid);
    GAUGE_SAVE<atype, Real>(arg.array, link, ids, DEVPARAMS::VolumeG * 4);
  } else {
    ids = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
               param_GridG(0) +
           x[0]) >>
          1;
    ids += arg.parity * param_HalfVolumeG();
    ids += mu * DEVPARAMS::VolumeG;
    // Load uplink from global memory
    msun link =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, ids, DEVPARAMS::VolumeG * 4);
    x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
    int idl = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
                   param_GridG(0) +
               x[0]) >>
              1;
    idl += (1 - arg.parity) * param_HalfVolumeG();
    idl += mu * DEVPARAMS::VolumeG;
    // Load downlink from global memory
    msun link1 =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, idl, DEVPARAMS::VolumeG * 4);
    // Gauge fix hits
    if (FUNCTIONTYPE == 3)
      GaugeFixHit_NoAtomicAdd<DIR, NTPVOL, Real>(link, link1, arg.relax_boost,
                                                 tid);
    if (FUNCTIONTYPE == 4)
      GaugeFixHit_AtomicAdd<DIR, NTPVOL, Real>(link, link1, arg.relax_boost,
                                               tid);
    if (FUNCTIONTYPE == 5)
      GaugeFixHit_NoAtomicAdd_LessSM<DIR, NTPVOL, Real>(link, link1,
                                                        arg.relax_boost, tid);
    GAUGE_SAVE<atype, Real>(arg.array, link, ids, DEVPARAMS::VolumeG * 4);
    GAUGE_SAVE<atype, Real>(arg.array, link1, idl, DEVPARAMS::VolumeG * 4);
  }
#ifdef ACTIVATE_COVARIANT_GAUGE_FIX_STR
  if (threadIdx.x < NTPVOL)
    arg.state[idr] = localState;
#endif
}

template <int DIR, bool UseTex, ArrayType atype, class Real>
class GaugeFix_Interior : Tunable {
private:
  string functionName;
  gauge array;
  double timesec;
  int grid[4];
  GaugeFixArg<Real> arg;
#ifdef TIMMINGS
  Timer mtime;
#endif
  dim3 createGrid(const dim3 &block) const {
    return Call_GaugeFix_createGrid(block, minThreads());
  }
  bool advanceBlockDim(TuneParam &param) const {
    return Call_GaugeFix_advanceBlockDim(param, minThreads(), sizeof(Real));
  }

private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const {
    return CalcSharedBytesPerBlock(param.block, sizeof(Real));
  }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.size; }

public:
  virtual void initTuneParam(TuneParam &param) const {
    CalcInitTuneParam(param.block, param.grid, param.shared_bytes, minThreads(),
                      sizeof(Real));
  }
  void Init() {
    arg.size = 1;
    for (int i = 0; i < 4; i++) {
      grid[i] = PARAMS::Grid[i];
      arg.size *= PARAMS::Grid[i];
    }
    arg.size = arg.size >> 1;
    timesec = 0.0;
#ifdef MULTI_GPU
    if (numnodes() > 1) {
      arg.size = 1;
      for (int d = 0; d < 4; d++) {
        if (comm_dim_partitioned(d)) {
          arg.border[d] = param_border(d) + 1;
          arg.X[d] = (param_GridG(d) - 2 * arg.border[d]);
          arg.size *= (param_GridG(d) - 2 * arg.border[d]);
        } else {
          arg.border[d] = 0;
          arg.X[d] = param_Grid(d);
          arg.size *= param_Grid(d);
        }
      }
      arg.size = arg.size >> 1;
    }
#endif
    arg.array = array.GetPtr();
    arg.parity = 0;
  }

  GaugeFix_Interior(gauge &array, Real relax_boost) : array(array) {
    functionName = "OVR_GaugeFix_Interior";
    arg.relax_boost = relax_boost;
    Init();
  }

  ~GaugeFix_Interior() {};
  /** Sets default values for when tuning is disabled - this is guaranteed to
   * work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const {
    initTuneParam(param);
  }
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << grid[0] << "x";
    vol << grid[1] << "x";
    vol << grid[2] << "x";
    vol << grid[3];
    aux << "threads=" << arg.size << ",prec=" << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(),
                   array.ToStringArrayType().c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << ","
       << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { array.Backup(); }
  void postTune() { array.Restore(); }
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef MULTI_GPU
    if (numnodes() == 1) {
      LAUNCH_KERNEL_GFIX(kernel_do_hit_EO_shared_combo, tp, stream, arg, DIR,
                         UseTex, atype, Real);
    } else {
      LAUNCH_KERNEL_GFIX(kernel_do_hit_EO_shared_combo_interior, tp, stream,
                         arg, DIR, UseTex, atype, Real);
    }
#else
    LAUNCH_KERNEL_GFIX(kernel_do_hit_EO_shared_combo, tp, stream, arg, DIR,
                       UseTex, atype, Real);
#endif
  }

  void Run(const cudaStream_t &stream, int parityin) {
#ifdef TIMMINGS
    mtime.start();
#endif
    arg.parity = parityin;
#ifdef MULTI_GPU
    StartExchange_gauge_fix_links_gauge<Real>(array, parityin);
#endif
    apply(stream);
#ifdef MULTI_GPU
    EndExchange_gauge_fix_links_gauge<Real>(array, parityin);
#endif
    CUDA_SAFE_DEVICE_SYNC();
#ifdef TIMMINGS
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
  }
  double time() { return timesec; }

  void stat() {
    COUT << functionName << ":  " << time() << " s\t" << bandwidth()
         << " GB/s\t" << flops() << " GFlops" << endl;
  }

  long long flop() const {
    // using the flops from GaugeFixHit_NoAtomicAdd with 8 threads per lattice
    // site...
    return (TOTAL_SUB_BLOCKS * (22LL + 22LL * DIR + 224LL * NCOLORS) +
            8LL * array.getNumFlop(true) + 8LL * array.getNumFlop(false)) *
           arg.size * numnodes();
  }
  long long bytes() const {
    return 8LL * 2 * array.getNumParams() * arg.size * sizeof(Real) *
           numnodes();
  }
  double flops() { return ((double)flop() * 1.0e-9) / timesec; }

  double bandwidth() { return (double)bytes() / (timesec * (double)(1 << 30)); }
};

template <class Real> struct GaugeFixBorderArg {
  complex *array;
  int *BorderPoints;
  int size;
  Real relax_boost;
  int parity;
  int nlinksfaces;
};

template <int FUNCTIONTYPE, int NTPVOL, int DIR, bool UseTex, ArrayType atype,
          class Real>
__global__ void
kernel_do_hit_EO_shared_combo_border(GaugeFixBorderArg<Real> arg) {
  // Get the local thread id for each site, since at each site we are assigning
  // 8 threads
  int tid = (threadIdx.x + NTPVOL) % NTPVOL;
  // Get the global thread id
#if (__CUDA_ARCH__ >= 300)
  int ids = blockIdx.x * NTPVOL + tid;
#else
  int ids = gridDim.x * blockIdx.y + blockIdx.x;
  ids = NTPVOL * ids + tid;
#endif
  if (ids >= arg.size)
    return;
  int mu = (threadIdx.x / NTPVOL);
  ids = arg.BorderPoints[ids + arg.parity * arg.nlinksfaces];
  if (ids > DEVPARAMS::Volume)
    return;
  int x[4];
  x[3] = ids / (DEVPARAMS::Grid[0] * DEVPARAMS::Grid[1] * DEVPARAMS::Grid[2]);
  x[2] = (ids / (DEVPARAMS::Grid[0] * DEVPARAMS::Grid[1])) % DEVPARAMS::Grid[2];
  x[1] = (ids / DEVPARAMS::Grid[0]) % DEVPARAMS::Grid[1];
  x[0] = ids % DEVPARAMS::Grid[0];

  for (int i = 0; i < 4; i++)
    x[i] += param_border(i);
  if (FUNCTIONTYPE < 3) {
    int parity = arg.parity;
    if (threadIdx.x >= NTPVOL * 4) {
      mu -= 4;
      x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
      parity = 1 - parity;
    }
    ids = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
               param_GridG(0) +
           x[0]) >>
          1;
    ids += parity * DEVPARAMS::HalfVolumeG;
    ids += mu * DEVPARAMS::VolumeG;
    // Load uplink and downlink from global memory
    msun link =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, ids, DEVPARAMS::VolumeG * 4);
    // Gauge fix hits
    if (FUNCTIONTYPE == 0)
      GaugeFixHit_NoAtomicAdd<DIR, NTPVOL, Real>(link, arg.relax_boost, tid);
    if (FUNCTIONTYPE == 1)
      GaugeFixHit_AtomicAdd<DIR, NTPVOL, Real>(link, arg.relax_boost, tid);
    if (FUNCTIONTYPE == 2)
      GaugeFixHit_NoAtomicAdd_LessSM<DIR, NTPVOL, Real>(link, arg.relax_boost,
                                                        tid);
    GAUGE_SAVE<atype, Real>(arg.array, link, ids, DEVPARAMS::VolumeG * 4);
  } else {
    ids = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
               param_GridG(0) +
           x[0]) >>
          1;
    ids += arg.parity * param_HalfVolumeG();
    ids += mu * DEVPARAMS::VolumeG;
    // Load uplink from global memory
    msun link =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, ids, DEVPARAMS::VolumeG * 4);
    x[mu] = (x[mu] - 1 + param_GridG(mu)) % param_GridG(mu);
    int idl = ((((x[3] * param_GridG(2) + x[2]) * param_GridG(1)) + x[1]) *
                   param_GridG(0) +
               x[0]) >>
              1;
    idl += (1 - arg.parity) * param_HalfVolumeG();
    idl += mu * DEVPARAMS::VolumeG;
    // Load downlink from global memory
    msun link1 =
        GAUGE_LOAD<UseTex, atype, Real>(arg.array, idl, DEVPARAMS::VolumeG * 4);
    // Gauge fix hits
    if (FUNCTIONTYPE == 3)
      GaugeFixHit_NoAtomicAdd<DIR, NTPVOL, Real>(link, link1, arg.relax_boost,
                                                 tid);
    if (FUNCTIONTYPE == 4)
      GaugeFixHit_AtomicAdd<DIR, NTPVOL, Real>(link, link1, arg.relax_boost,
                                               tid);
    if (FUNCTIONTYPE == 5)
      GaugeFixHit_NoAtomicAdd_LessSM<DIR, NTPVOL, Real>(link, link1,
                                                        arg.relax_boost, tid);
    GAUGE_SAVE<atype, Real>(arg.array, link, ids, DEVPARAMS::VolumeG * 4);
    GAUGE_SAVE<atype, Real>(arg.array, link1, idl, DEVPARAMS::VolumeG * 4);
  }
}

static inline __device__ void LatticeFaceIndices(int &x1, int &x2, int &x3,
                                                 int &x4, int idd, int oddbit,
                                                 int faceid, int borderid) {
  int za, xodd;
  switch (faceid) {
  case 0: // X FACE
    za = idd / (DEVPARAMS::Grid[1] / 2);
    x4 = za / DEVPARAMS::Grid[2];
    x3 = za - x4 * DEVPARAMS::Grid[2];
    xodd = (borderid + x3 + x4 + oddbit) & 1;
    x2 = (2 * idd + xodd) - za * DEVPARAMS::Grid[1];
    x1 = borderid;
    break;
  case 1: // Y FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x4 = za / DEVPARAMS::Grid[2];
    x3 = za - x4 * DEVPARAMS::Grid[2];
    xodd = (borderid + x3 + x4 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x2 = borderid;
    break;
  case 2: // Z FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x4 = za / DEVPARAMS::Grid[1];
    x2 = za - x4 * DEVPARAMS::Grid[1];
    xodd = (borderid + x2 + x4 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x3 = borderid;
    break;
  case 3: // T FACE
    za = idd / (DEVPARAMS::Grid[0] / 2);
    x3 = za / DEVPARAMS::Grid[1];
    x2 = za - x3 * DEVPARAMS::Grid[1];
    xodd = (borderid + x2 + x3 + oddbit) & 1;
    x1 = (2 * idd + xodd) - za * DEVPARAMS::Grid[0];
    x4 = borderid;
    break;
  }
}

__global__ void COMPUTE_ALLFACEINDICES(int *faceindices, int facesize,
                                       int faceid, int oddbit) {
  int idd = INDEX1D();
  if (idd < facesize) {
    int x1, x2, x3, x4;
    int borderid = 0;
    int idx = idd;
    if (idx >= facesize / 2) {
      borderid = DEVPARAMS::Grid[faceid] - 1;
      idx -= facesize / 2;
    }
    LatticeFaceIndices(x1, x2, x3, x4, idx, oddbit, faceid, borderid);
    faceindices[idd] =
        (((x4 * DEVPARAMS::Grid[2] + x3) * DEVPARAMS::Grid[1]) + x2) *
            DEVPARAMS::Grid[0] +
        x1;
  }
}

template <int DIR, bool UseTex, ArrayType atype, class Real>
class GaugeFix_Border : Tunable {
private:
  string functionName;
  gauge array;
  double timesec;
  int grid[4];
  GaugeFixBorderArg<Real> arg;
#ifdef TIMMINGS
  Timer mtime;
#endif

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    if (numnodes() > 1)
      LAUNCH_KERNEL_GFIX(kernel_do_hit_EO_shared_combo_border, tp, stream, arg,
                         DIR, UseTex, atype, Real);
  }

  dim3 createGrid(const dim3 &block) const {
    return Call_GaugeFix_createGrid(block, minThreads());
  }
  bool advanceBlockDim(TuneParam &param) const {
    return Call_GaugeFix_advanceBlockDim(param, minThreads(), sizeof(Real));
  }

private:
  unsigned int sharedBytesPerThread() const { return 0; }

  unsigned int sharedBytesPerBlock(const TuneParam &param) const {
    return CalcSharedBytesPerBlock(param.block, sizeof(Real));
  }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.size; }

public:
  virtual void initTuneParam(TuneParam &param) const {
    CalcInitTuneParam(param.block, param.grid, param.shared_bytes, minThreads(),
                      sizeof(Real));
  }
  void Init() {
    arg.array = array.GetPtr();
    arg.parity = 0;
    arg.size = 1;
    timesec = 0.0;
    if (numnodes() > 1) {
      for (int i = 0; i < 4; i++)
        grid[i] = PARAMS::Grid[i];
      // Pre calculate lattice indices for halo and/or interior domain points
      arg.nlinksfaces = 0;
      for (int fc = 0; fc < PARAMS::NActiveFaces; fc++)
        arg.nlinksfaces += PARAMS::FaceSize[PARAMS::FaceId[fc]];

      arg.BorderPoints = (int *)dev_malloc(2 * arg.nlinksfaces * sizeof(int));
      thrust::device_ptr<int> array_faceT[2];
      for (int oddbit = 0; oddbit < 2; oddbit++) {
        array_faceT[oddbit] = thrust::device_pointer_cast(
            arg.BorderPoints + oddbit * arg.nlinksfaces);
      }
      dim3 threads(128, 1, 1);
      int start = 0;
      for (int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
        dim3 blocks =
            GetBlockDim(threads.x, PARAMS::FaceSize[PARAMS::FaceId[fc]]);
        if (fc > 0)
          start += PARAMS::FaceSize[PARAMS::FaceId[fc - 1]];
        for (int oddbit = 0; oddbit < 2; oddbit++)
          COMPUTE_ALLFACEINDICES<<<blocks, threads>>>(
              arg.BorderPoints + oddbit * arg.nlinksfaces + start,
              PARAMS::FaceSize[PARAMS::FaceId[fc]], PARAMS::FaceId[fc], oddbit);
      }
      int bordersize[2];
      for (int i = 0; i < 2; i++) {
        // sort and remove duplicated lattice indices
        thrust::sort(array_faceT[i], array_faceT[i] + arg.nlinksfaces);
        thrust::device_ptr<int> new_end =
            thrust::unique(array_faceT[i], array_faceT[i] + arg.nlinksfaces);
        bordersize[i] = thrust::raw_pointer_cast(new_end) -
                        thrust::raw_pointer_cast(array_faceT[i]);
      }
      arg.size = bordersize[0];
      if (bordersize[0] != bordersize[1])
        errorCULQCD("Error in thread size border....\n");
    }
  }
  GaugeFix_Border(gauge &array, Real relax_boost) : array(array) {
    functionName = "GaugeFix Border";
    arg.relax_boost = relax_boost;
    Init();
  }
  ~GaugeFix_Border() {
    if (numnodes() > 1) {
      dev_free(arg.BorderPoints);
    }
  };
  /** Sets default values for when tuning is disabled - this is guaranteed to
   * work, but will be slow */
  virtual void defaultTuneParam(TuneParam &param) const {
    initTuneParam(param);
  }
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << grid[0] << "x";
    vol << grid[1] << "x";
    vol << grid[2] << "x";
    vol << grid[3];
    aux << "threads=" << arg.size << ",prec=" << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(),
                   array.ToStringArrayType().c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << ","
       << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { array.Backup(); }
  void postTune() { array.Restore(); }
  void Run(const cudaStream_t &stream, int parityin) {
#ifdef TIMMINGS
    mtime.start();
#endif
    arg.parity = parityin;
    if (numnodes() > 1)
      apply(stream);
#ifdef TIMMINGS
    CUDA_SAFE_DEVICE_SYNC();
    mtime.stop();
    timesec = mtime.getElapsedTimeInSec();
#endif
  }
  double time() { return timesec; }

  void stat() {
    COUT << functionName << ":  " << time() << " s\t" << bandwidth()
         << " GB/s\t" << flops() << " GFlops" << endl;
  }

  long long flop() const {
    // using the flops from GaugeFixHit_NoAtomicAdd with 8 threads per lattice
    // site...
    return (TOTAL_SUB_BLOCKS * (22LL + 22LL * DIR + 224LL * NCOLORS) +
            8LL * array.getNumFlop(true) + 8LL * array.getNumFlop(false)) *
           arg.size * numnodes();
  }
  long long bytes() const {
    return 8LL * 2 * array.getNumParams() * arg.size * sizeof(Real) *
           numnodes();
  }

  double flops() { return ((double)flop() * 1.0e-9) / timesec; }

  double bandwidth() { return (double)bytes() / (timesec * (double)(1 << 30)); }
};
#endif

#endif
