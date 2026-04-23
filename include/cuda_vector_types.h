#ifndef CUDA_VECTOR_TYPES_H
#define CUDA_VECTOR_TYPES_H

/*
#ifdef __CUDACC__
#include <vector_functions.h>
#include <vector_types.h>
#else
struct __attribute__((aligned(8))) float2 {    float x, y; };
struct __attribute__((aligned(16))) float4 {    float x, y, z, w; };
struct __attribute__((aligned(16))) double2 {    double x, y; };
struct __attribute__((aligned(16))) double4 {    double x, y, z, w; };
#endif
*/
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <vector_types.h>

/* double4_16a / make_double4_16a exist from CUDA 13; keep double4 on older
 * toolkits to avoid missing-type build failures. */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
#define CULQCD_MAKE_DOUBLE4(x, y, z, w) make_double4_16a((x), (y), (z), (w))
#else
#define CULQCD_MAKE_DOUBLE4(x, y, z, w) make_double4((x), (y), (z), (w))
#endif

namespace CULQCD {

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
typedef double4_16a culqcd_double4;
#else
typedef double4 culqcd_double4;
#endif

template <typename Real, int number> struct MakeVector;
template <> struct MakeVector<float, 2> {
  typedef float2 type;
};
template <> struct MakeVector<double, 2> {
  typedef double2 type;
};
template <> struct MakeVector<float, 4> {
  typedef float4 type;
};
template <> struct MakeVector<double, 4> {
  typedef culqcd_double4 type;
};
} // namespace CULQCD
#endif // #ifndef CUDA_VECTOR_TYPES_H
