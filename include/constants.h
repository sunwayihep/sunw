#ifndef CONSTANTS_H_DEF
#define CONSTANTS_H_DEF

#include <complex.h>
#include <vector>

namespace CULQCD {

/*
DIRS:
    XUP = 0
    YUP = 1
    ZUP = 2
    TUP = 3
*/

#define mod(x, y) ((x) % (y))
#define pow2(x) ((x) * (x)) // pow(x, 2)
// #define PI	3.1415926535897932
// #define PII	6.2831853071795865
#ifndef PI
#define PI 3.1415926535897932384626433832795 // pi
#endif
#ifndef PII
#define PII 6.2831853071795864769252867665590 // 2 * pi
#endif

// the total number of spatial and time plaqs per lattice sites
#define TOTAL_NUM_PLAQS ((NDIMS * (NDIMS - 1)) / 2)
// total number of time component plaqs per lattice sites
#define TOTAL_NUM_TPLAQS (NDIMS - 1)
// total number of spatial component plaqs per lattice sites
#define TOTAL_NUM_SPLAQS (TOTAL_NUM_PLAQS - TOTAL_NUM_TPLAQS)

/**
    @brief Macro to copy a variable to GPU constant memory
    @param dev Device parameter to set
    @param host Host parameter to copy
    @param type parameter type, (int, float, double, ....)
*/
#define memcpyToSymbol(dev, host, type)                                        \
  CUDA_SAFE_CALL(                                                              \
      cudaMemcpyToSymbol(dev, host, sizeof(type), 0, cudaMemcpyHostToDevice));

/**
    @brief Macro to copy an array to GPU constant memory
    @param dev Device array to set
    @param host Host array to copy
    @param type array type, (int, float, double, ....)
    @param length array length
*/
#define memcpyToArraySymbol(dev, host, type, length)                           \
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev, host, length * sizeof(type), 0,       \
                                    cudaMemcpyHostToDevice));

/*! @brief HOST Lattice parameters.*/
namespace PARAMS {
/*! \brief If true uses texture memory instead of global memory */
extern bool UseTex;

extern double Beta;
extern std::vector<int> lattice_size;
/*! \brief Number of lattice points in X, Y, Z, T direction */
extern int NX, NY, NZ, NT;
/*! \brief Number of lattice points on this node in X, Y, Z, T direction */
// extern	int	 	nx, ny, nz, nt;
/*! \brief Size = nx x ny */
extern int kstride;
/*! \brief Size = nx x ny x nz */
extern int tstride;
/*! \brief Size = nx x ny x nz x nt */
// extern	int	 	volume;
/*! \brief Size of the total gauge field, nx x ny x nz x nt x 4 */
extern int size;

/*! \brief Mesh coordinates of this node */
extern int logical_coordinate[NDIMS];
/*! \brief Number of nodes per lattice dimension, machine dimensions */
extern int Grid[NDIMS];
extern int GridWGhost[NDIMS];
extern int Volume;
extern int HalfVolume;
extern int VolumeG;
extern int HalfVolumeG;
/*! \brief Face size for each dimension */
extern int FaceSize[NDIMS];
extern int FaceSizeG[NDIMS];
// extern bool     activeFace[4];
extern int NActiveFaces;
extern int FaceId[NDIMS];
// Node id +1 for active faces
extern int NodeIdRight[NDIMS];
// Node id -1 for active faces
extern int NodeIdLeft[NDIMS];
// #endif
/*! \brief store the max gpu grid size in x dimension */
extern uint GPUGridDimX;

extern int Border[NDIMS];

extern cudaDeviceProp deviceProp;

extern dim3 nthreadsPHB;
extern dim3 nthreadsOVR;
extern dim3 nblocksPHB;
extern dim3 nblocksOVR;
extern dim3 nthreadsPLAQ;
extern dim3 nblocksPLAQ;
extern dim3 nthreadsINIT;
extern dim3 nblocksINIT;
extern dim3 nthreadsINITHALF;
extern dim3 nblocksINITHALF;
extern dim3 nthreadsREU;
extern dim3 nblocksREU;
} // namespace PARAMS

// double reportPotentialOccupancy(void *kernel, int blockSize, size_t
// dynamicSMem);

/**
    @brief Calculate the number of blocks in 1D thread block.
    If the number of blocks is bigger than the x gridDim dimension, then it uses
   a 2D gridDim.
    In Fermi GPUs, the max GridDim.x is 65535 which can be easily exhausted.
*/
dim3 GetBlockDim(size_t threads, size_t size);

void PrintDetails();

/**
    @brief Set the global lattice parameters. This function also copies these
   global parameters to GPU constant memory, void copyConstantsToGPU();
*/
void SETPARAMS(bool _usetex, double beta, int nx, int ny, int nz, int nt,
               bool verbose);

void SETPARAMS(bool _usetex, double beta, std::vector<int> lattice_size,
               bool verbose);

void SETPARAMS(bool _usetex, int latticedim[4], const int nodesperdim[4],
               const int logical_coordinate[4], bool verbose);

/**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate the index's, the total number of blocks is
   NCOLORS * ( NCOLORS - 1) / 2.
    @return Returns two index's in int2 type, accessed by .x and .y.
*/
__host__ __device__ inline int2 IndexBlock(int block) {
  int2 id;
  int i1;
  int found = 0;
  int del_i = 0;
  int index = -1;
  while (del_i < (NCOLORS - 1) && found == 0) {
    del_i++;
    for (i1 = 0; i1 < (NCOLORS - del_i); i1++) {
      index++;
      if (index == block) {
        found = 1;
        break;
      }
    }
  }
  id.y = i1 + del_i;
  id.x = i1;
  return id;
}
/**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate de index's, the total number of blocks is
   NCOLORS * ( NCOLORS - 1) / 2.
    @param p store the first index
    @param q store the second index
*/
__host__ __device__ inline void IndexBlock(int block, int &p, int &q) {
#if (NCOLORS == 3)
  if (block == 0) {
    p = 0;
    q = 1;
  } else if (block == 1) {
    p = 1;
    q = 2;
  } else {
    p = 0;
    q = 2;
  }
#else
  int i1;
  int found = 0;
  int del_i = 0;
  int index = -1;
  while (del_i < (NCOLORS - 1) && found == 0) {
    del_i++;
    for (i1 = 0; i1 < (NCOLORS - del_i); i1++) {
      index++;
      if (index == block) {
        found = 1;
        break;
      }
    }
  }
  q = i1 + del_i;
  p = i1;
#endif
}
/**
    @brief Prints to screen the global lattice parameters and the currently GPU
   memory usage.
*/
void Details();

/*! @brief GPU/DEVICE Lattice parameters, must be copied to GPU constant memory.
 */
namespace DEVPARAMS {
/*! \brief If true uses texture memory instead of global memory */
extern __constant__ bool UseTex;
/*! \brief inverse of the gauge coupling, beta = 2 Nc / g_0^2 */
extern __constant__ double Beta;
/*! \brief inverse of the gauge coupling over Nc, betaOverNc = 2 / g_0^2 */
extern __constant__ double BetaOverNc;
/*! \brief Size = Nx x Ny */
extern __constant__ int kstride;
/*! \brief Size = Nx x Ny x Nz */
extern __constant__ int tstride;
/*! \brief Size of the total gauge field, Nx x Ny x Nz x Nt x 4 */
extern __constant__ int size;

extern __constant__ int Grid[NDIMS];
extern __constant__ int GridWGhost[NDIMS];
/*! \brief Size = Nx x Ny x Nz x Nt */
extern __constant__ int Volume;
/*! \brief Size = Nx x Ny x Nz x Nt / 2 */
extern __constant__ int HalfVolume;
extern __constant__ int VolumeG;
extern __constant__ int HalfVolumeG;
extern __constant__ int Border[NDIMS];
/*! \brief HYP smearing constant: alpha1 */
extern __constant__ float hypalpha1;
/*! \brief HYP smearing constant: alpha2 */
extern __constant__ float hypalpha2;
/*! \brief HYP smearing constant: alpha3 */
extern __constant__ float hypalpha3;
} // namespace DEVPARAMS

int __host__ __device__ inline param_border(int i) {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::Border[i];
#else
  return PARAMS::Border[i];
#endif
}

bool __host__ __device__ inline param_Tex() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::UseTex;
#else
  return PARAMS::UseTex;
#endif
}
double __host__ __device__ inline param_Beta() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::Beta;
#else
  return PARAMS::Beta;
#endif
}
int __host__ __device__ inline param_Volume() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::Volume;
#else
  return PARAMS::Volume;
#endif
}
int __host__ __device__ inline param_VolumeG() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::VolumeG;
#else
  return PARAMS::VolumeG;
#endif
}
int __host__ __device__ inline param_HalfVolume() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::HalfVolume;
#else
  return PARAMS::HalfVolume;
#endif
}
int __host__ __device__ inline param_HalfVolumeG() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::HalfVolumeG;
#else
  return PARAMS::HalfVolumeG;
#endif
}
int __host__ __device__ inline param_Grid(int dim) {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::Grid[dim];
#else
  return PARAMS::Grid[dim];
#endif
}
int __host__ __device__ inline param_GridG(int dim) {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::GridWGhost[dim];
#else
  return PARAMS::GridWGhost[dim];
#endif
}
int __host__ __device__ inline param_Size() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::size;
#else
  return PARAMS::size;
#endif
}
int __host__ __device__ inline param_SizeG() {
#ifdef __CUDA_ARCH__
  return DEVPARAMS::size * 4;
#else
  return PARAMS::size * 4;
#endif
}

/**
    @brief Initialize constants in constant GPU memory. Copy the contents of the
   constants defined in PARAMS namespace.
*/
void copyConstantsToGPU();
void copyConstantsToGPU0();
/**
    @brief Allows to turn ON/OFF the use of Textures.
    @param TexOn if true turn on reads from texture.
*/
void UseTextureMemory(bool TexOn);
/**
    @brief Set constants for the HYP smearing.
*/
void copyHYPSmearConstants(float _alpha1, float _alpha2, float _alpha3);
} // namespace CULQCD
#endif
