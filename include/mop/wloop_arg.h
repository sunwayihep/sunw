#pragma once
namespace CULQCD{
template<class Real>
struct WLOPArg{
  complex *gaugefield;
  complex *fieldOp;
  int radius;
  int mu;
  int opN;
};
}
