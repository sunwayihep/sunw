
#ifndef GAUGEFIXING_H
#define GAUGEFIXING_H

#include <gaugearray.h>
#include <typeinfo>

#include <timer.h>
#include <tune.h>

namespace CULQCD {

/**
   @brief Apply Landau/Coulomb Gauge Fixing with Overrelaxation method. Only
   supports evenodd arrays.
   @param array gauge field to be fixed
   @param DIR DIR=4 for Landau gauge Fixing and DIR=3 for Coulomb gauge fixing
   @param relax_boost Overrelaxation parameter
   @param stopvalue criterium to stop the method, precision
   @param maxsteps maximum number of iterations
   @param reunit_interval Reunitarize when iteration count is a multiple of this
   @param verbose check convergence and print to screen Fg and theta when
   iteration count is a multiple of this
   @return last value of the theta parameter
*/
template <class Real>
complex GaugeFixingOvr(gauge _pgauge, int DIR, Real relax_boost, Real stopvalue,
                       int maxsteps, int reunit_interval, int verbose);

/**
   @brief Apply Landau/Coulomb Gauge Fixing with Steepest Descent Method with
   Fourier Acceleration.
   @param array gauge field to be fixed
   @param DIR DIR=4 for Landau gauge Fixing and DIR=3 for Coulomb gauge fixing
   @param alpha constant for the method, optimal value is 0.08
   @param landautune if true auto tune method
   @param stopvalue criterium to stop the method, precision
   @param maxsteps maximum number of iterations/sweeps
   @param verbose check convergence and print to screen Fg and theta when
   iteration count is a multiple of this
   @param useGx if true pre-calculates g(x), therefore using more Device memory
   but gives better performance
   @param atypeGx array type for g(x), SOA/SOA12 for SU(3) and SOA for SU(N>3)
   @return last value of the theta parameter
*/
template <class Real>
complex GaugeFixingFFT(gauge _pgauge, int DIR, Real alpha, bool landautune,
                       Real stopvalue, int maxsteps, int verbose,
                       ArrayType atypeDeltax = SOA, bool useGx = false,
                       ArrayType atypeGx = SOA);

// For array stored in normal index order, not even/odd order
namespace NormalIdOrder {
template <class Real>
complex GaugeFixingFFT(gauge _pgauge, int DIR, Real alpha, bool landautune,
                       Real stopvalue, int maxsteps, int verbose,
                       ArrayType atypeDeltax = SOA, bool useGx = false,
                       ArrayType atypeGx = SOA);
}

namespace NormalIdOrder1 {
template <class Real>
complex GaugeFixingFFT(gauge _pgauge, int DIR, Real alpha, bool landautune,
                       Real stopvalue, int maxsteps, int verbose,
                       ArrayType atypeDeltax = SOA, bool useGx = false,
                       ArrayType atypeGx = SOA);
}

/**
   @brief Measure the gauge fixing quality, Fg and theta.
   @param DIR DIR=4 for Landau gauge Fixing and DIR=3 for Coulomb gauge fixing
   @param UseTex if true uses texture memory
   @param atype gauge array type, types supported: SOA/SOA12/SOA for SU(3) and
   SOA for SU(N>3)
   @param Real precision of the gauge array, double/float

   mode to use, example:

   GaugeFixQuality<4, false, SOA, double> quality(gauge_array);
   quality.Run(); -> call kernel to calculate Fg and theta, also return the
   value of Fg and theta as a complex pair
   quality.printValue(); -> print value of Fg and theta to screen
   quality.stat(); -> performance status
*/
template <int DIR, bool UseTex, ArrayType atype, class Real>
class GaugeFixQuality : Tunable {
private:
  string functionName;
  gauge array;
  complex *sum;
  int size;
  complex value;
  double timesec;
  int grid[4];
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
  GaugeFixQuality(gauge &array);
  ~GaugeFixQuality();
  complex Run(const cudaStream_t &stream);
  complex Run();
  double time();
  void stat();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  void printValue();
  complex Value() const { return value; }
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
};

template <class Real> struct GaugeFixQualityArg {
  complex *array;
  complex *value;
};

/**
   @brief Measure the gauge fixing quality, Fg and theta, reduction using CUDA
   CUB, uses less device memory
   @param DIR DIR=4 for Landau gauge Fixing and DIR=3 for Coulomb gauge fixing
   @param UseTex if true uses texture memory
   @param atype gauge array type, types supported: SOA/SOA12/SOA for SU(3) and
   SOA for SU(N>3)
   @param Real precision of the gauge array, double/float

   mode to use, example:

   GaugeFixQualityCUB<4, false, SOA, double> quality(gauge_array);
   quality.Run(); -> call kernel to calculate Fg and theta, also return the
   value of Fg and theta as a complex pair
   quality.printValue(); -> print value of Fg and theta to screen
   quality.stat(); -> performance status
*/
template <int DIR, bool UseTex, ArrayType atype, class Real>
class GaugeFixQualityCUB : Tunable {
private:
  string functionName;
  gauge array;
  GaugeFixQualityArg<Real> arg;
  int size;
  complex value;
  double timesec;
  int grid[4];
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
  GaugeFixQualityCUB(gauge &array);
  ~GaugeFixQualityCUB();
  complex Run(const cudaStream_t &stream);
  complex Run();
  double time();
  void stat();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  void printValue();
  complex Value() const { return value; }
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
};
} // namespace CULQCD
#endif