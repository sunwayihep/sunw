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
#include <vector_functions.h>
#include <vector_types.h>

namespace CULQCD{

template <typename Real, int number> struct MakeVector;
template <> struct MakeVector<float, 2>
{
    typedef float2 type;
};
template <> struct MakeVector<double, 2>
{
    typedef double2 type;
};
template <> struct MakeVector<float, 4>
{
    typedef float4 type;
};
template <> struct MakeVector<double, 4>
{
    typedef double4 type;
};

}
#endif // #ifndef CUDA_VECTOR_TYPES_H
