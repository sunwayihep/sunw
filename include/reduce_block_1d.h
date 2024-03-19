
#ifndef __REDUCE_BLOCK_1D_H__
#define __REDUCE_BLOCK_1D_H__


#include <cudaAtomic.h>
#include <sharedmemtypes.h>


#define WARP_SIZE 32

namespace CULQCD{

template<typename T>
inline __device__ void reduce_block_1d(T *ptr, const T &thread_val){

	SharedMemory<T> smem;

	smem[threadIdx.x] = thread_val;
	__syncthreads();

	//Only one active warp do the reduction
	//Bocks are always multiple of the warp size!
	if(blockDim.x > WARP_SIZE && threadIdx.x < WARP_SIZE){
		for(uint s = 1; s < blockDim.x / WARP_SIZE; s++)
			smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + WARP_SIZE * s]; 
	}
	//__syncthreads(); //No need to synchronize inside warp!!!!
	//One thread do the warp reduction
	if(threadIdx.x == 0 ) {
		T sum = 0.;
		for(uint s = 0; s < WARP_SIZE; s++) 
			sum += smem[s];
		CudaAtomicAdd(ptr, sum);
	}
}
//DON'T FORGET to reserve the shared memory need by this in the kernel parameters:
//                     <<<,,threads_per_block * sizeof(T),>>>




//#define CUDA_SHFL_DOWN(val, offset) __shfl_down(val, offset)
#define FULL_MASK 0xffffffff
#define CUDA_SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)


template<typename T>
inline __device__ T cuda_shfl_down(T val, int offset){
	return CUDA_SHFL_DOWN(val, offset);
}
template<>
inline __device__ complexd cuda_shfl_down(complexd val, int offset){
	val.real() = CUDA_SHFL_DOWN(val.real(), offset);
	val.imag() = CUDA_SHFL_DOWN(val.imag(), offset);
	return val;
}
template<>
inline __device__ complexs cuda_shfl_down(complexs val, int offset){
	val.real() = CUDA_SHFL_DOWN(val.real(), offset);
	val.imag() = CUDA_SHFL_DOWN(val.imag(), offset);
	return val;
}


template<typename T>
inline __device__ T warpReduceSum(T val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
	val += cuda_shfl_down(val, offset);
  return val;
}






template<typename T>
inline __device__ T blockReduceSum(T val) {

  __shared__ T shared[WARP_SIZE]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}




template<typename T>
inline __device__ void reduce_block_1d_shfl(T *ptr, const T &thread_val){

	T sum = blockReduceSum<T>(thread_val);
	if(threadIdx.x == 0 ) CudaAtomicAdd(ptr, sum);
}
//DON'T FORGET to reserve the shared memory need by this in the kernel parameters:
//  




}

#endif

