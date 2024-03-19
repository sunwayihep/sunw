


#include <texture.h>


namespace CULQCD{
//////////////////////////////////////////////////////////////////////////////////////
/*
  TEXTURES
*/
//////////////////////////////////////////////////////////////////////////////////////
texture<float2, 1, cudaReadModeElementType> tex_gauge_float;
texture<int4, 1, cudaReadModeElementType> tex_gauge_double;

texture<float2, 1, cudaReadModeElementType> tex_gx_float;
texture<int4, 1, cudaReadModeElementType> tex_gx_double;

texture<float2, 1, cudaReadModeElementType> tex_delta_float;
texture<int4, 1, cudaReadModeElementType> tex_delta_double;

texture<float2, 1, cudaReadModeElementType> tex_lambda_float;
texture<int4, 1, cudaReadModeElementType> tex_lambda_double;



}