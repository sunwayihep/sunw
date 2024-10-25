
#ifndef TEXTURE_H
#define TEXTURE_H

#include <cuda.h>
#include <complex.h>

namespace CULQCD {
//////////////////////////////////////////////////////////////////////////////////////
/*
  TEXTURES
*/
//////////////////////////////////////////////////////////////////////////////////////
#define DEFINE_TEXTURE(type, name, id)                                         \
  texture<type, 1, cudaReadModeElementType> name##_a##id
////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern texture<float2, 1, cudaReadModeElementType> tex_gauge_float;
extern texture<int4, 1, cudaReadModeElementType> tex_gauge_double;
////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern texture<float2, 1, cudaReadModeElementType> tex_gx_float;
extern texture<int4, 1, cudaReadModeElementType> tex_gx_double;
////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern texture<float2, 1, cudaReadModeElementType> tex_delta_float;
extern texture<int4, 1, cudaReadModeElementType> tex_delta_double;
////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern texture<float2, 1, cudaReadModeElementType> tex_lambda_float;
extern texture<int4, 1, cudaReadModeElementType> tex_lambda_double;
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// READ TEXTURES
//////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_GAUGE(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_GAUGE<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_gauge_float, id));
}
template <> inline __device__ complexd TEXTURE_GAUGE<double>(uint id) {
  int4 u = tex1Dfetch(tex_gauge_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_GAUGE_CONJ(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_GAUGE_CONJ<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_gauge_float, id)).conj();
}
template <> inline __device__ complexd TEXTURE_GAUGE_CONJ<double>(uint id) {
  int4 u = tex1Dfetch(tex_gauge_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), -__hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_DELTA(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_DELTA<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_delta_float, id));
}
template <> inline __device__ complexd TEXTURE_DELTA<double>(uint id) {
  int4 u = tex1Dfetch(tex_delta_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_DELTA_CONJ(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_DELTA_CONJ<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_delta_float, id)).conj();
}
template <> inline __device__ complexd TEXTURE_DELTA_CONJ<double>(uint id) {
  int4 u = tex1Dfetch(tex_delta_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), -__hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_GX(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_GX<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_gx_float, id));
}
template <> inline __device__ complexd TEXTURE_GX<double>(uint id) {
  int4 u = tex1Dfetch(tex_gx_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_GX_CONJ(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_GX_CONJ<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_gx_float, id)).conj();
}
template <> inline __device__ complexd TEXTURE_GX_CONJ<double>(uint id) {
  int4 u = tex1Dfetch(tex_gx_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), -__hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> inline __device__ complex TEXTURE_LAMBDA(uint id) {
  return complex::zero();
}
template <> inline __device__ complexs TEXTURE_LAMBDA<float>(uint id) {
  return make_complexs(tex1Dfetch(tex_lambda_float, id));
}
template <> inline __device__ complexd TEXTURE_LAMBDA<double>(uint id) {
  int4 u = tex1Dfetch(tex_lambda_double, id);
  return make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

#endif // #ifndef TEXTURE_H
