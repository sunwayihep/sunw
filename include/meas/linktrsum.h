
#ifndef LINKTRSUM_H
#define LINKTRSUM_H

#include <gaugearray.h>
#include <tune.h>
#include <timer.h>

namespace CULQCD {

/////////////////////////////////////////////////////////////////////////////////////////
//////// Gauge Trace
/////////////////////////////////////////////////////////////////////////////////////////
template <class Real> struct TraceArg {
  complex *array;
  complex *value;
};

//#ifdef USE_CUDA_CUB
template <class Real> class GaugeTraceCUB : Tunable {
private:
  gauge array;
  TraceArg<Real> arg;
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
  GaugeTraceCUB(gauge &array);
  ~GaugeTraceCUB();
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
  TuneKey tuneKey() const;
  std::string paramString(const TuneParam &param) const;
  void preTune() {}
  void postTune() {}
};
//#else
template <class Real> class GaugeTrace : Tunable {
private:
  gauge array;
  TraceArg<Real> arg;
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
  // sum is a device array with lattice volume size
  // must be allocated before calling this!!!!!
  GaugeTrace(gauge &array, complex *sum);
  ~GaugeTrace();
  complex Run(const cudaStream_t &stream);
  complex Run();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();
  void printValue();
  TuneKey tuneKey() const;
  std::string paramString(const TuneParam &param) const;
  void preTune() {}
  void postTune() {}
};
//#endif
}

#endif
