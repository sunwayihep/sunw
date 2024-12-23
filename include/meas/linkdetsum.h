
#ifndef LINKDETSUM_H
#define LINKDETSUM_H

#include <gaugearray.h>

#include <timer.h>
#include <tune.h>

namespace CULQCD {

/////////////////////////////////////////////////////////////////////////////////////////
//////// Gauge determinant
/////////////////////////////////////////////////////////////////////////////////////////
template <class Real> struct DetArg {
  complex *array;
  complex *value;
};

// #ifdef USE_CUDA_CUB
template <class Real> class GaugeDetCUB : Tunable {
private:
  gauge array;
  DetArg<Real> arg;
  int size;
  complex value;
  double timesec;
#ifdef TIMMINGS
  Timer mtime;
#endif
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream);

public:
  GaugeDetCUB(gauge &array);
  ~GaugeDetCUB();
  complex Run(const cudaStream_t &stream);
  complex Run();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();
  void printValue();
  complex Value() const { return value; }
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
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
  void preTune() {}
  void postTune() {}
};
// #else
template <class Real> class GaugeDet : Tunable {
private:
  gauge array;
  DetArg<Real> arg;
  int size;
  complex value;
  double timesec;
#ifdef TIMMINGS
  Timer mtime;
#endif
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream);

public:
  GaugeDet(gauge &array);
  ~GaugeDet();
  complex Run(const cudaStream_t &stream);
  complex Run();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();
  void printValue();
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
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
  void preTune() {}
  void postTune() {}
};
// #endif
} // namespace CULQCD

#endif
