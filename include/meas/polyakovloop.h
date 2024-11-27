

#ifndef POLYAKOVLOOP_H
#define POLYAKOVLOOP_H

#include <gaugearray.h>
#include <matrixsun.h>
#include <typeinfo>

#include <timer.h>
#include <tune.h>

namespace CULQCD {

template <class Real> class OnePolyakovLoop : Tunable {
private:
  string functionName;
  typedef void (*TFuncPtr)(complex *, complex *);
  TFuncPtr kernel_pointer;
  gauge array;
  complex *sum;
  complex *tmp;
  int size;
  complex poly_value;
  double timesec;
  bool tex;
  int grid[NDIMS];
#ifdef TIMMINGS
  Timer mtime;
#endif

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return size; }
  void apply(const cudaStream_t &stream);
  void SetFunctionPtr();

public:
  OnePolyakovLoop(gauge &array);
  ~OnePolyakovLoop();

  complex Run(const cudaStream_t &stream);
  complex Run();
  double flops();
  double bandwidth();
  long long flop() const;
  long long bytes() const;
  double time();
  void stat();
  void printValue();
  complex Value() const { return poly_value; }

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

template <class Real>
void Calculate_OnePloyakovLoop(gauge array, gauge ploop, bool savePLMatrix);
template <class Real>
void Calculate_TrPloyakovLoop(gauge array, complex *ploop);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////        Calculate one polyakov loop/mean
//////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
   @brief Calculate the mean polyakov loop
   @param array gauge field
   @return complex mean polyakov loop of the current gauge field
*/
template <class Real> complex PolyakovLoop(gauge array);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////        Calculate Two polyakov loops
//////////////////////////////////
///////////////////////////        Color Avg. Free Energy/ Singlet free Energy
//////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
   @brief Calculate the Color average free energy (real part) and singlet free
   energy (imaginary part) and prints the result to screen
   @param array gauge field
   @param dist  maximum distance between the sources, [0,dist]
*/
template <class Real> void TwoPolyakovLoop(gauge array, int dist);
/**
   @brief Calculate the color average free energy (real part) and singlet free
   energy (imaginary part)
   @param array gauge field
   @param dist  maximum distance between the sources, [0,dist]
   @return complex array with size = dist+1, real part is for color average free
   energy and the imaginary part for the singlet free energy
*/
template <class Real> complex *GetTwoPolyakovLoop(gauge array, int dist);

template <class Real> void PolyakovLoop3D(gauge array, gauge ploop);

template <class Real> complex GetTwoPolyakovLoop(gauge ploop, int3 dist);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////        Calculate Two polyakov loops
//////////////////////////////////
///////////////////////////                   Singlet free Energy
//////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
   @brief Calculate the singlet free energy and prints the result to screen
   @param array gauge field
   @param dist  maximum distance between the sources, [0,dist]
*/
template <class Real> void SingletFreeEnergy(gauge array, int dist);
/**
   @brief Calculate the singlet free energy
   @param array gauge field
   @param dist  maximum distance between the sources, [0,dist]
   @return complex array with size = dist+1, complex singlet free energy
*/
template <class Real> complex *GetSingletFreeEnergy(gauge array, int dist);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////        Calculate Two polyakov loops
//////////////////////////////////
///////////////////////////              Color Avg. Free Energy
//////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
   @brief Calculate the color average free energy and prints the result to
   screen
   @param array gauge field
   @param lat lattice definitions, including copying constant to GPU memory and
   thread and block sizes
   @param dist  maximum distance between the sources, [0,dist]
*/
template <class Real> void ColorAvgFreeEnergy(gauge array, int dist);
/**
   @brief Calculate the color average free energy
   @param array gauge field
   @param dist  maximum distance between the sources, [0,dist]
   @return complex array with size = dist+1, complex color average free energy
*/
template <class Real> complex *GetColorAvgFreeEnergy(gauge array, int dist);
} // namespace CULQCD

#endif
