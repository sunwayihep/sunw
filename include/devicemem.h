

#ifndef DEVICEMEM_H
#define DEVICEMEM_H

#include <cuda.h>
#include <cuda_runtime.h>


namespace CULQCD{
/**
	@brief Display current GPU memory usage
*/
void CheckMemUsage();
/**
	@brief Returns a unigned int3 with current memory usage, .x for used memory, .y for free memory and .z for GPU total memory
*/
uint3 GetMemUsage();
/**
	@brief Display available GPUs in the system as well as the current memory usage for each one
*/
void GPUsInSystem();


void GPUDetails();
}

#endif // #ifndef DEVICEMEM_H
