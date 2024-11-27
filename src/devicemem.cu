

#include <cuda_common.h>
#include <devicemem.h>
#include <iostream>

using namespace std;

namespace CULQCD {

/**
        @brief Display current GPU memory usage
*/
void CheckMemUsage() {
  size_t mfree, total;
  int gpuid = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
  CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &total));
  float freemem = (float)mfree / (1024.0 * 1024.0);
  float totalmem = (float)total / (1024.0 * 1024.0);
  float usedmem = totalmem - freemem;
  cout << "Currently Memory Usage (used/free/total) on GPU(" << gpuid
       << "): " << usedmem << "/" << freemem << "/" << totalmem << " MB "
       << endl;
}

/**
        @brief Returns a unigned int3 with current memory usage, .x for used
   memory, .y for free memory and .z for GPU total memory
*/
uint3 GetMemUsage() {
  size_t mfree, mtotal, mused;
  int gpuid = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
  CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
  mused = mtotal - mfree;
  return make_uint3(mused, mfree, mtotal);
}

/**
        @brief Display available GPUs in the system as well as the current
   memory usage for each one
*/
void GPUsInSystem() {
  std::cout << "---------------------------------------------------------------"
               "---------------------"
            << std::endl;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }
  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
    printf("There are no available device(s) that support CUDA\n");
  else
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    CUDA_SAFE_CALL(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    size_t mfree, total;
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &total));
    float freemem = (float)mfree / (1024.0 * 1024.0);
    float totalmem = (float)total / (1024.0 * 1024.0);
    float usedmem = totalmem - freemem;
    cout << "  Currently Memory Usage (used/free/total) on GPU(" << dev
         << "): " << usedmem << "/" << freemem << "/" << totalmem << " MB "
         << endl;
  }
  cout << endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------"
            << std::endl;
}

/**
        @brief Display available GPUs in the system as well as the current
   memory usage for each one
*/
void GPUDetails() {
  std::cout << "---------------------------------------------------------------"
               "---------------------"
            << std::endl;
  int dev = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&dev));
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
         deviceProp.major, deviceProp.minor);
  size_t mfree, total;
  CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &total));
  float freemem = (float)mfree / (1024.0 * 1024.0);
  float totalmem = (float)total / (1024.0 * 1024.0);
  float usedmem = totalmem - freemem;
  cout << "  Currently Memory Usage (used/free/total) on GPU(" << dev
       << "): " << usedmem << "/" << freemem << "/" << totalmem << " MB "
       << endl;
  cout << endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------"
            << std::endl;
}

} // namespace CULQCD