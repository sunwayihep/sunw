#ifndef PLAQUETTE_H
#define PLAQUETTE_H

#include <complex.h>
#include <gaugearray.h>
#include <timer.h>
#include <typeinfo>

#include <launch_kernel.cuh>
#include <tune.h>

namespace CULQCD {

template <class Real>
void PlaquetteFieldSpace(gauge array, complex *plaq, complex *meanplaq);

template <class Real>
void PlaquetteField(gauge array, complex *plaq, complex *meanplaq);

template <class Real> class Plaquette : Tunable {
private:
  typedef void (*TFuncPtr)(complex *, complex *);
  TFuncPtr kernel_pointer;
  gauge array;
  complex *sum;
  int size;
  complex plaq_value;
  double timesec;
  int grid[4];
  bool reduced;
  bool tex;
#ifdef TIMMINGS
  Timer plaqtime;
#endif

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream);
  void SetFunctionPtr();

public:
  Plaquette(gauge &array, complex *sum);
  Plaquette() {};
  ~Plaquette() {};

  void Run(const cudaStream_t &stream, bool calcmeanvalue);
  void Run(bool calcmeanvalue);
  complex Reduce(const cudaStream_t &stream);
  complex Reduce();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();
  void printValue();
  complex Value();

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << grid[0] << "x";
    vol << grid[1] << "x";
    vol << grid[2] << "x";
    vol << grid[3];
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

  // Return the minimum size of complex* sum array in bytes....
  static size_t ASumBytes() { return PARAMS::Volume * sizeof(complex); }
};

template <class Real> struct PlaqArg {
  complex *pgauge;
  complex *plaq;
};

template <class Real> class PlaquetteCUB : Tunable {
private:
  string functionName;
  PlaqArg<Real> arg;
  gauge array;
  int size;
  complex plaq_value;
  double timesec;
  int numparams;
  int grid[4];
  string atype;
#ifdef TIMMINGS
  Timer plaqtime;
#endif

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream);

public:
  PlaquetteCUB(gauge &array);
  ~PlaquetteCUB() { dev_free(arg.plaq); };
  complex Run(const cudaStream_t &stream);
  complex Run();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();
  void printValue();
  complex Value() const { return plaq_value; }

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
  void preTune() {}
  void postTune() {}
};
} // namespace CULQCD

#endif
