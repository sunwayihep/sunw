

#ifndef REDUCE_T2_OP_CUH
#define REDUCE_T2_OP_CUH

inline __host__ __device__ int2 operator+(int2 u, int2 v) {
  return make_int2(u.x + v.x, u.y + v.y);
}

inline __host__ __device__ void operator+=(int2 &a, int2 b) {
  a.x += b.x;
  a.y += b.y;
}
inline __host__ __device__ void operator+=(volatile int2 &a, volatile int2 b) {
  a.x += b.x;
  a.y += b.y;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 operator+(float2 u, float2 v) {
  return make_float2(u.x + v.x, u.y + v.y);
}

inline __host__ __device__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}
inline __host__ __device__ void operator+=(volatile float2 &a,
                                           volatile float2 b) {
  a.x += b.x;
  a.y += b.y;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ double2 operator+(double2 u, double2 v) {
  return make_double2(u.x + v.x, u.y + v.y);
}

inline __host__ __device__ void operator+=(double2 &a, double2 b) {
  a.x += b.x;
  a.y += b.y;
}
inline __host__ __device__ void operator+=(volatile double2 &a,
                                           volatile double2 b) {
  a.x += b.x;
  a.y += b.y;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
/*inline __host__ __device__ int4 operator+( int4 u, int4 v){
  return make_int4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
  }

  inline __host__ __device__ void operator+=( int4 &a, int4 b ) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  }
  inline __host__ __device__ void operator+=( volatile int4 &a, volatile int4 b
  ) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
  }*/
//////////////////////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float4 operator+(float4 u, float4 v) {
  return make_float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__ void operator+=(volatile float4 &a,
                                           volatile float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ double4 operator+(double4 u, double4 v) {
  return make_double4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

inline __host__ __device__ void operator+=(double4 &a, double4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__ void operator+=(volatile double4 &a,
                                           volatile double4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
//////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> inline __host__ __device__ T zero() {
  T res;
  return res;
}

// INT
template <> inline __host__ __device__ int zero<int>() { return 0; }
template <> inline __host__ __device__ int2 zero<int2>() {
  return make_int2(0, 0);
}
template <> inline __host__ __device__ int4 zero<int4>() {
  return make_int4(0, 0, 0, 0);
}

// FLOAT
template <> inline __host__ __device__ float zero<float>() {
  return (float)0.0;
}
template <> inline __host__ __device__ float2 zero<float2>() {
  return make_float2((float)0.0, (float)0.0);
}
template <> inline __host__ __device__ float4 zero<float4>() {
  return make_float4((float)0.0, (float)0.0, (float)0.0, (float)0.0);
}

// DOUBLE
template <> inline __host__ __device__ double zero<double>() {
  return (double)0.0;
}
template <> inline __host__ __device__ double2 zero<double2>() {
  return make_double2((double)0.0, (double)0.0);
}
template <> inline __host__ __device__ double4 zero<double4>() {
  return make_double4((double)0.0, (double)0.0, (double)0.0, (double)0.0);
}

#endif
