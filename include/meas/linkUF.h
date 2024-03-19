
#ifndef LINKUF_H
#define LINKUF_H

#include <gaugearray.h>

#include <tune.h>
#include <timer.h>

namespace CULQCD{

/////////////////////////////////////////////////////////////////////////////////////////
//////// Gauge determinant
/////////////////////////////////////////////////////////////////////////////////////////
template<class Real>
struct GaugeUFArg{
  complex *array;
  complex *value;
};

template <class Real> 
class GaugeUFCUB: Tunable{
private:
   gauge array;
   GaugeUFArg<Real> arg;
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
   GaugeUFCUB(gauge &array);
   ~GaugeUFCUB();
   complex Run(const cudaStream_t &stream);
   complex Run();
   double flops();
   double bandwidth();
   long long flop() const ;
   long long bytes() const;
   double time();
   void stat();
   void printValue();
   complex Value()const{return value;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    return TuneKey(vol.str().c_str(), typeid(*this).name(), array.ToStringArrayType().c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() {  }
};









}

#endif

