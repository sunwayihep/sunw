
#ifndef RANDOM_GPU_H
#define RANDOM_GPU_H

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <curand_kernel.h>


#include <alloc.h>
#include <comm_mpi.h>

namespace CULQCD{

#if defined(XORWOW)
typedef struct curandStateXORWOW cuRNGState;
#elif defined(MRG32k3a)
typedef struct curandStateMRG32k3a cuRNGState;
#else
typedef struct curandStateMRG32k3a cuRNGState;
#endif




struct randArg{
    #ifdef MULTI_GPU
    int log_cord[4]; //logical node coordinate
    int grid[4]; //global lattice dimensions
    int X[4];//local lattice dimensions
    #endif
    /*! @brief number of curand states */
    int size;
    /*! initial rng seed */
    unsigned int seed;
};

/**
    @brief Class declaration to initialize and hold CURAND RNG states
*/
class RNG {
public:
    /*! array with current curand rng state */
    cuRNGState *state;
    cuRNGState *backup_state;
    randArg arg;
    bool backup_Indev;
    RNG();
    /*! initialize curand rng states with seed */
    void Init(unsigned int seedin);
    /*! free array */
    void Release(); 
    /*! @brief return curand rng array size */
    int& Size(){ return arg.size;};
    /*! @brief return curand rng array initialseed */
    unsigned int& Seed(){ return arg.seed;};
    /*! @brief return curand rng array of states */
    cuRNGState* State(){ return state;};

    size_t Bytes(){
        return arg.size * sizeof(cuRNGState);
    }
    void Backup(){
        printfCULQCD("Backup RNG...\n");
        size_t mfree, mtotal;
        int gpuid=-1;
        CUDA_SAFE_CALL(cudaGetDevice(&gpuid));
        CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
        if(Bytes() < mfree){
            backup_Indev = true;
            backup_state = (cuRNGState*) dev_malloc(Bytes());
            CUDA_SAFE_CALL(cudaMemcpy(backup_state, state, Bytes(), cudaMemcpyDeviceToDevice));
        } 
        else{
            backup_Indev = false;
            backup_state = (cuRNGState*) safe_malloc(Bytes());
            CUDA_SAFE_CALL(cudaMemcpy(backup_state, state, Bytes(), cudaMemcpyDeviceToHost));
        }
    }
    void Restore(){
        printfCULQCD("Restore RNG...\n");
        if(backup_Indev){
            CUDA_SAFE_CALL(cudaMemcpy(state, backup_state, Bytes(), cudaMemcpyDeviceToDevice));
            dev_free(backup_state);
        } 
        else{
            CUDA_SAFE_CALL(cudaMemcpy(state, backup_state, Bytes(), cudaMemcpyHostToDevice));
            host_free(backup_state);
        }
    }
    void Save(std::string filename);
    void Read(std::string filename);
private:
    /*! @brief allocate curand rng states array in device memory */
    void AllocateRNG();
    /*! @brief CURAND array states initialization */
    void INITRNG();


};





/**
   @brief Return a random number between a and b
   @param state curand rng state
   @param a lower range
   @param b upper range
   @return  random number in range a,b
*/
template<class Real>
inline  __device__ Real Random(cuRNGState &state, Real a, Real b){
    Real res;
    return res;
}
 
template<>
inline  __device__ float Random<float>(cuRNGState &state, float a, float b){
    return a + (b - a) * curand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state, double a, double b){
    return a + (b - a) * curand_uniform_double(&state);
}

/**
   @brief Return a random number between 0 and 1
   @param state curand rng state
   @return  random number in range 0,1
*/
template<class Real>
inline  __device__ Real Random(cuRNGState &state){
    Real res;
    return res;
}
 
template<>
inline  __device__ float Random<float>(cuRNGState &state){
    return curand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state){
    return curand_uniform_double(&state);
}



template<class Real>
inline  __device__ Real RandomNormal(cuRNGState &state){
    Real res;
    return res;
}

template<>
inline  __device__ float RandomNormal<float>(cuRNGState &state){
    return curand_normal(&state);
}

template<>
inline  __device__ double RandomNormal<double>(cuRNGState &state){
    return curand_normal_double(&state);
}



template<class Real>
struct uniform { };
template<>
struct uniform<float> {
    __device__
        static inline float rand(cuRNGState &state) {
        return curand_uniform(&state);
    }
};
template<>
struct uniform<double> {
    __device__
        static inline double rand(cuRNGState &state) {
        return curand_uniform_double(&state);
    }
};



template<class Real>
struct normal { };
template<>
struct normal<float> {
    __device__
        static inline float rand(cuRNGState &state) {
        return curand_normal(&state);
    }
};
template<>
struct normal<double> {
    __device__
        static inline double rand(cuRNGState &state) {
        return curand_normal_double(&state);
    }
};

}


#endif 
