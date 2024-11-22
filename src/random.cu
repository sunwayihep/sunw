
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include <cuda_common.h>
#include <random.h>
#include <constants.h>
#include <cuda.h>

#include <comm_mpi.h>
#include <alloc.h>


#include <index.h>

using namespace std;

namespace CULQCD{



/**
    @brief CUDA kernel to initialize CURAND RNG states
    @param state CURAND RNG state array
    @param seed initial seed for RNG
    @param arg arguments for single (size) and multi-GPU size (size, global lattice dimensions 
    local lattice dimensions and logical node coordinate)
*/
__global__ void 
kernel_random(cuRNGState *state, randArg arg ){
#if (__CUDA_ARCH__ >= 300)
	int id = blockIdx.x * blockDim.x + threadIdx.x;
#else
	int id = gridDim.x * blockIdx.y  + blockIdx.x;
	id = blockDim.x * id + threadIdx.x; 
#endif
    if(id >= arg.size) return;
    #ifdef MULTI_GPU
    int x[4];
    Index_4D_EO(x, id, 0, DEVPARAMS::Grid);
    for(int i=0; i<4;i++) x[i] += arg.log_cord[i] * arg.X[i];
    int idd = ((((x[3] * arg.grid[2] + x[2]) * arg.grid[1]) + x[1] ) * arg.grid[0] + x[0]) >> 1 ;
    curand_init(arg.seed, idd, 0, &state[id]);
    #else
    curand_init(arg.seed, id, 0, &state[id]);
    #endif
}

/**
    @brief Call CUDA kernel to initialize CURAND RNG states
    @param state CURAND RNG state array
    @param seed initial seed for RNG
    @param rng_size size of the CURAND RNG state array
    @param offset this parameter is used to skip ahead the index in the sequence, usefull for multigpu. 
*/
void launch_kernel_random(cuRNGState *state, randArg arg){  
    dim3 nthreads(128,1,1); //Put this in auto-tune?!?!?!?!?
    dim3 nblocks = GetBlockDim(nthreads.x, arg.size);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig( kernel_random,   cudaFuncCachePreferL1));
    kernel_random<<<nblocks,nthreads>>>(state, arg);
    CUDA_SAFE_DEVICE_SYNC( );
    CUT_CHECK_ERROR("Initialize RNG: Kernel execution failed");
    COUT << "Array of RNG states initialized with:\n\tsize: " << arg.size << "\n\tinitial seed: " << arg.seed << std::endl;
    //std::cout << "Array of RNG states initialized with:\n\tsize: " << rng_size << "\n\tinitial seed: " << seed << "\n\toffset: "<< offset  << std::endl;
}

RNG::RNG(){
    state = NULL;
    arg.seed = 0;
    arg.size = 1;
    for(int d=0; d < NDIMS; d++){
        #ifdef MULTI_GPU
        arg.log_cord[d] = PARAMS::logical_coordinate[d];
        arg.X[d] = PARAMS::Grid[d];
        #endif
        arg.size *= PARAMS::Grid[d];
    }
	//std::cout << PARAMS::Grid[3] << "::::" << arg.log_cord[0] <<"\t" << arg.log_cord[1] <<"\t" << arg.log_cord[2] <<"\t" << arg.log_cord[3]<< std::endl;
    arg.size = arg.size >> 1;
    #ifdef MULTI_GPU
    arg.grid[0] = PARAMS::NX;
    arg.grid[1] = PARAMS::NY;
    arg.grid[2] = PARAMS::NZ;
    arg.grid[3] = PARAMS::NT;
    #endif
}   
            
/**
    @brief Initialize CURAND RNG states
*/
void RNG::Init(unsigned int seedin){
    arg.seed = seedin;
    INITRNG();
}   


/**
    @brief Initialize CURAND RNG states
*/
void RNG::INITRNG(){
	if(state == NULL) {
		if(arg.size > 0){
			AllocateRNG();
		}
		else{
            errorCULQCD("Array of random numbers not allocated, array size: %d!\nExiting...\n", arg.size);
		}
	}
	if(state != NULL){
        int size = 1;
        for(int d=0; d < NDIMS; d++) size *= PARAMS::Grid[d];
        size = size >> 1;
        if(size != arg.size){
            arg.size = 1;
            for(int d=0; d < NDIMS; d++){
                #ifdef MULTI_GPU
                arg.log_cord[d] = PARAMS::logical_coordinate[d];
                arg.X[d] = PARAMS::Grid[d];
                #endif
                arg.size *= PARAMS::Grid[d];
            }
            arg.size = arg.size >> 1;
            #ifdef MULTI_GPU
            arg.grid[0] = PARAMS::NX;
            arg.grid[1] = PARAMS::NY;
            arg.grid[2] = PARAMS::NZ;
            arg.grid[3] = PARAMS::NT;
            #endif
            Release();
            AllocateRNG();
        }
	}
	launch_kernel_random(state, arg);
}		

/**
    @brief Allocate Device memory for CURAND RNG states
*/
void RNG::AllocateRNG(){
    state = (cuRNGState*)dev_malloc(arg.size * sizeof(cuRNGState));
    COUT << "Allocated array of random numbers with rng_size: " << Bytes()/(float)(1048576) << " MB" << std::endl;
}
/**
    @brief Release Device memory for CURAND RNG states
*/
void RNG::Release(){
    if(state != NULL){
        dev_free(state);
        COUT << "Free array of random numbers with rng_size: " << Bytes()/(float)(1048576) << " MB" << std::endl;
        arg.size = 0;
        state = NULL;
    }
}








void RNG::Save(string filename){
    if(numnodes() > 1) errorCULQCD("Not implemented YET in multi-GPU mode...\n");

    cuRNGState *hoststate = (cuRNGState*) safe_malloc(Bytes());
    CUDA_SAFE_CALL(cudaMemcpy(hoststate, state, Bytes(), cudaMemcpyDeviceToHost));

    ofstream fileout;
    fileout.open(filename.c_str(), ios::binary | ios::out);
    if (!fileout.is_open()){
        errorCULQCD("Error saving rng state...\n");
    }
    COUT << "Saving RNG state to file: " << filename << endl;
    fileout.write((const char*)hoststate, arg.size * sizeof(cuRNGState));
    if ( fileout.fail() ) {
        errorCULQCD("ERROR: Unable to write file: %s\n", filename.c_str());
    }
    fileout.close();
    host_free(hoststate);
}
void RNG::Read(string filename){
    if(numnodes() > 1) errorCULQCD("Not implemented YET in multi-GPU mode...\n");

    cuRNGState *hoststate = (cuRNGState*) safe_malloc(Bytes());

    ifstream filein;
    filein.open(filename.c_str(), ios::binary | ios::in);
    if (!filein.is_open()){
        errorCULQCD("Error reading rng state...\n");
    }
    COUT << "Reading RNG state to file: " << filename << endl;
    filein.read((char*)hoststate, arg.size * sizeof(cuRNGState));
    if ( filein.fail() ) {
        errorCULQCD("ERROR: Unable read file: %s\n", filename.c_str());
    }
    filein.close();
    CUDA_SAFE_CALL(cudaMemcpy(state, hoststate, Bytes(), cudaMemcpyHostToDevice));
    host_free(hoststate);
}



}
