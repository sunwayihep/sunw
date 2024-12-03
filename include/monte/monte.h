
#ifndef MONTE_H
#define MONTE_H

#include <gaugearray.h>
#include <random.h>
#include <typeinfo>

#include <timer.h>
#include <tune.h>

namespace CULQCD {

template <class Real> class HeatBath : Tunable {
private:
  typedef void (*TFuncPtr)(complex *, cuRNGState *, int, int);
  TFuncPtr kernel_pointer;
  gauge array;
  RNG randstates;
  int size;
  double timesec;
  int grid[NDIMS];
  bool tex;
#ifdef TIMMINGS
  Timer mtime;
#endif
  int dir, parity;

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream);
  void SetFunctionPtr();

public:
  HeatBath(gauge &array, RNG &randstates);
  ~HeatBath() {};

  void Run(const cudaStream_t &stream);
  void Run();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    for (int i = 0; i < NDIMS - 1; i++)
      vol << grid[i] << "x";
    vol << grid[NDIMS - 1];
    aux << "threads=" << size << ",prec=" << sizeof(Real);
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
  void preTune() {
    array.Backup();
    randstates.Backup();
  }
  void postTune() {
    array.Restore();
    randstates.Restore();
  }
};
} // namespace CULQCD

#endif
