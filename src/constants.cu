

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_common.h>
#include <constants.h>
#include <comm_mpi.h>
#include <exchange.h>


namespace CULQCD{

#define RADIUS_BORDER 2
//Radius border for the multi-GPU gauge
//MUST BE MULTIPLE OF 2

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace PARAMS{
    /*! \brief If true uses texture memory instead of global memory */
    bool	UseTex = false;
    double	Beta = 6.2;
    /*! \brief Size = Nx x Ny */
    int	 	kstride = 0;
    /*! \brief Size = Nx x Ny x Nz */
    int	 	tstride = 0;
    /*! \brief Size = Nx x Ny x Nz x Nt */
    int	 	volume = 0;
    /*! \brief Size of the total gauge field, Nx x Ny x Nz x Nt x 4 */
    int		size = 0;
	std::vector<int> lattice_size;
	int 	NX, NY, NZ, NT;


    int      Grid[NDIMS];
    int      GridWGhost[NDIMS];
    int 	 Volume;
    int 	 VolumeG;
    int 	 HalfVolume;
    int 	 HalfVolumeG;
    int	 	Border[NDIMS];

	int logical_coordinate[NDIMS];
	int FaceSize[NDIMS];
	int FaceSizeG[NDIMS];
	//bool activeFace[4];

	//Number of active faces to exchange borders, 4 faces, means exchanges in X,Y,Z and T
	int NActiveFaces = 0;
	int FaceId[NDIMS];

    /*! @brief GPU grid size in x dimension */
    uint GPUGridDimX = 0;

	cudaDeviceProp deviceProp;

	int NodeIdRight[NDIMS];
	int NodeIdLeft[NDIMS];



    dim3 nthreadsPHB(0,0,0);
    dim3 nthreadsOVR(0,0,0);
    dim3 nblocksPHB(0,0,0);
    dim3 nblocksOVR(0,0,0);
    dim3 nthreadsPLAQ(0,0,0);
    dim3 nblocksPLAQ(0,0,0);
    dim3 nthreadsINIT(0,0,0);
    dim3 nblocksINIT(0,0,0);
    dim3 nthreadsINITHALF(0,0,0);
    dim3 nblocksINITHALF(0,0,0);
    dim3 nthreadsREU(0,0,0);
    dim3 nblocksREU(0,0,0);

}


//FOR NOW STAYS HERE....
/*double reportPotentialOccupancy(void *kernel, int blockSize, size_t dynamicSMem){
    int device;
    cudaDeviceProp prop;

    int numBlocks;
    int activeWarps;
    int maxWarps;

    double occupancy;

    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));

    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &numBlocks,
                        kernel,
                        blockSize,
                        dynamicSMem));

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    occupancy = (double)activeWarps / maxWarps;

    return occupancy;
}**/


dim3 GetBlockDim(size_t threads, size_t size){
	uint blockx = BLOCKSDIVUP(size, threads);
	uint blocky = 1;
	if(blockx > PARAMS::GPUGridDimX){
		blocky = BLOCKSDIVUP(blockx, PARAMS::GPUGridDimX);
		blockx = PARAMS::GPUGridDimX;
	}
	dim3 blocks(blockx,blocky,1);
	return blocks;
}



void PrintDetails(){
	COUT << "----------------------------------------------------------------" << std::endl;
	COUT << "Nd: " << NDIMS << std::endl;
	COUT << "Nc: " << NCOLORS << std::endl;
	COUT << "beta: " << PARAMS::Beta << std::endl;
	COUT << "Lattice Dimensions: ";
	for(auto i: PARAMS::lattice_size) COUT << i << ' ';
	COUT << std::endl; 
	
	#ifdef MULTI_GPU
	COUT << "Local Lattice Dimension: " << PARAMS::Grid[0] << "x" <<PARAMS::Grid[1]<<"x"<<PARAMS::Grid[2]<<"x"<<PARAMS::Grid[3]<<std::endl;
	COUT << "Local Lattice Dimensions with ghost links: " \
	 << PARAMS::GridWGhost[0] << "x" \
	 << PARAMS::GridWGhost[1] << "x" \
	 << PARAMS::GridWGhost[2] << "x" \
	 << PARAMS::GridWGhost[3] << std::endl;
	COUT << "Number of active Faces:\t" << PARAMS::NActiveFaces;
	char facename[5]="XYZT";
	if(PARAMS::NActiveFaces>0) COUT << "\t->\t{";
	for(int fc = 0; fc < PARAMS::NActiveFaces; fc++) {
	    COUT << facename[PARAMS::FaceId[fc]];
	    if(fc < PARAMS::NActiveFaces - 1) COUT << ",";
	}
	if(PARAMS::NActiveFaces > 0) COUT << "}";
	COUT << std::endl;
	#endif
	COUT << "----------------------------------------------------------------" << std::endl;
}


void SETPARAMS(bool _usetex, double beta, int nx, int ny, int nz, int nt, bool verbose){
	/*static bool setparams = false;
	if(setparams){
		COUT << "Local parameters already set... Nothing to do here..." << std::endl;
		return;
	}*/
	PARAMS::Beta = beta;

	COUT << "Setting up lattice parameters..." << std::endl;
	// Set Host parameters
	PARAMS::UseTex = _usetex;
	setup_hyper_prime(nx,ny,nz,nt);
	PARAMS::NX = nx;
	PARAMS::NY = ny;
	PARAMS::NZ = nz;
	PARAMS::NT = nt;
	PARAMS::Grid[0] = nx;
	PARAMS::Grid[1] = ny;
	PARAMS::Grid[2] = nz;
	PARAMS::Grid[3] = nt;
	////////////////////////////////////////////
	for(int i = 0; i < 4; i++) 
		if( ((PARAMS::Grid[i] / nodes_per_dim(i)) % 2) != 0){
			COUT << "GPU code does not support odd lattice dimensions." << std::endl;
			exit(1);
		}
	for(int i = 0; i < 4; i++) PARAMS::Grid[i] /= nodes_per_dim(i);
	logical_coordinate(PARAMS::logical_coordinate);


	PARAMS::NActiveFaces = 0;
	for(int i = 0; i < 4; i++){
		if(nodes_per_dim(i) > 1 ) {
			PARAMS::Border[i] = RADIUS_BORDER;
			PARAMS::GridWGhost[i] = PARAMS::Grid[i] + 2 * PARAMS::Border[i];
			PARAMS::FaceId[PARAMS::NActiveFaces] = i;
			PARAMS::NActiveFaces++;
		}
		else {
			PARAMS::GridWGhost[i] = PARAMS::Grid[i];
			PARAMS::Border[i] = 0;
		}
	}
	PARAMS::FaceSize[0] = PARAMS::Grid[1] * PARAMS::Grid[2] * PARAMS::Grid[3];
	PARAMS::FaceSize[1] = PARAMS::Grid[0] * PARAMS::Grid[2] * PARAMS::Grid[3];
	PARAMS::FaceSize[2] = PARAMS::Grid[0] * PARAMS::Grid[1] * PARAMS::Grid[3];
	PARAMS::FaceSize[3] = PARAMS::Grid[0] * PARAMS::Grid[1] * PARAMS::Grid[2];
	PARAMS::FaceSizeG[0] = PARAMS::GridWGhost[1] * PARAMS::GridWGhost[2] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[1] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[2] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[2] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[1] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[3] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[1] * PARAMS::GridWGhost[2];

//MPI_Barrier(MPI_COMM_WORLD);
	/*printf("---->>>>>>>%d::%d:::::%d:%d:%d:%d:::::%d:%d:%d:%d\n",mynode(), numnodes(),PARAMS::GridWGhost[0],PARAMS::GridWGhost[1],\
		PARAMS::GridWGhost[2],PARAMS::GridWGhost[3],param_border(0),param_border(1),param_border(2),param_border(3));
	printf("---->>>>>>>%d::%d:::::%d:%d:%d:%d:::::%d:%d:%d:%d\n",mynode(), numnodes(),PARAMS::logical_coordinate[0],PARAMS::logical_coordinate[1],\
		PARAMS::logical_coordinate[2],PARAMS::logical_coordinate[3],nodes_per_dim(0),nodes_per_dim(1),nodes_per_dim(2),nodes_per_dim(3));*/
//MPI_Barrier(MPI_COMM_WORLD);
	int temp[4];
	for(int i = 0; i < 4; i++) temp[i] = PARAMS::logical_coordinate[i];
	for(int fc = 0; fc < PARAMS::NActiveFaces; fc++){
		int i = PARAMS::FaceId[fc];
		temp[i] = (PARAMS::logical_coordinate[i] + 1) % nodes_per_dim(i);
		PARAMS::NodeIdRight[fc] = temp[0] + temp[1] * nodes_per_dim(0);
		PARAMS::NodeIdRight[fc] += temp[2] * nodes_per_dim(0) * nodes_per_dim(1);
		PARAMS::NodeIdRight[fc] += temp[3] * nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2);
		temp[i] = PARAMS::logical_coordinate[i];

		temp[i] = (PARAMS::logical_coordinate[i] - 1 + nodes_per_dim(i)) % nodes_per_dim(i);
		PARAMS::NodeIdLeft[fc] = temp[0] + temp[1] * nodes_per_dim(0);
		PARAMS::NodeIdLeft[fc] += temp[2] * nodes_per_dim(0) * nodes_per_dim(1);
		PARAMS::NodeIdLeft[fc] += temp[3] * nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2);
		temp[i] = PARAMS::logical_coordinate[i];

		//printf("##%d:::%d:%d:%d:%d\n",mynode(), fc,PARAMS::NodeIdLeft[fc],PARAMS::NodeIdRight[fc],i);
	}

	PARAMS::kstride = PARAMS::Grid[0] * PARAMS::Grid[1];
	PARAMS::tstride = PARAMS::kstride * PARAMS::Grid[2];    

	PARAMS::Volume = 1;
	PARAMS::VolumeG = 1;
	for(int i = 0; i < 4; i++){
		PARAMS::Volume *= PARAMS::Grid[i];
		PARAMS::VolumeG *= PARAMS::GridWGhost[i];
	}
	PARAMS::HalfVolume = PARAMS::Volume / 2;
	PARAMS::HalfVolumeG = PARAMS::VolumeG / 2;
	PARAMS::size = PARAMS::Volume  * 4;


	if(verbose) PrintDetails();
	//Set Device parameters
	//Copy to GPU constant memory
	copyConstantsToGPU();
	int dev;
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	cudaGetDeviceProperties(&PARAMS::deviceProp, dev);
	PARAMS::GPUGridDimX = PARAMS::deviceProp.maxGridSize[0];

	 


	PARAMS::nthreadsPHB = dim3(64,1,1);
	PARAMS::nblocksPHB = GetBlockDim(PARAMS::nthreadsPHB.x, PARAMS::HalfVolume);
	PARAMS::nthreadsOVR = dim3(128,1,1);
	PARAMS::nblocksOVR = GetBlockDim(PARAMS::nthreadsOVR.x, PARAMS::HalfVolume);
	PARAMS::nthreadsINIT = dim3(128,1,1); 
	PARAMS::nblocksINIT = GetBlockDim(PARAMS::nthreadsINIT.x, PARAMS::Volume);
	PARAMS::nthreadsINITHALF = dim3(128,1,1); 
	PARAMS::nblocksINITHALF = GetBlockDim(PARAMS::nthreadsINITHALF.x, PARAMS::HalfVolume);
	PARAMS::nthreadsREU = dim3(128,1,1);
	PARAMS::nblocksREU = GetBlockDim(PARAMS::nthreadsREU.x, PARAMS::size);
	PARAMS::nthreadsPLAQ = dim3(64,1,1); //best 32^4 -> 64, next was for 128
	PARAMS::nblocksPLAQ =GetBlockDim(PARAMS::nthreadsPLAQ.x, PARAMS::Volume);


	//setparams = true;
	#ifdef MULTI_GPU
	FreeTempBuffersAndStreams();
	#endif
}


void SETPARAMS(bool _usetex, double beta, std::vector<int> lattice_size, bool verbose){
	PARAMS::Beta = beta;
	PARAMS::lattice_size = lattice_size;

	COUT << "Setting up lattice parameters..." << std::endl;
	// Set Host parameters
	PARAMS::UseTex = _usetex;
	if(lattice_size.size() < 2){
		std::cerr << "Number of space-time dimension NDIMS = "<<lattice_size.size()<< ", should no less than 2"<<std::endl;
		exit(0);
	}
	//setup_hyper_prime(nx,ny,nz,nt);
	//PARAMS::NX = nx;
	//PARAMS::NY = ny;
	//PARAMS::NZ = nz;
	//PARAMS::NT = nt;
	for(int i=0; i<NDIMS; i++) PARAMS::Grid[i] = lattice_size[i];
	//PARAMS::Grid[0] = nx;
	//PARAMS::Grid[1] = ny;
	//PARAMS::Grid[2] = nz;
	//PARAMS::Grid[3] = nt;
	////////////////////////////////////////////
	for(int i = 0; i < NDIMS; i++) 
		if( ((PARAMS::Grid[i] / nodes_per_dim(i)) % 2) != 0){
			COUT << "GPU code does not support odd lattice dimensions." << std::endl;
			exit(1);
		}
	for(int i = 0; i < NDIMS; i++) PARAMS::Grid[i] /= nodes_per_dim(i);
	logical_coordinate(PARAMS::logical_coordinate);


	PARAMS::NActiveFaces = 0;
	for(int i = 0; i < NDIMS; i++){
		if(nodes_per_dim(i) > 1 ) {
			PARAMS::Border[i] = RADIUS_BORDER;
			PARAMS::GridWGhost[i] = PARAMS::Grid[i] + 2 * PARAMS::Border[i];
			PARAMS::FaceId[PARAMS::NActiveFaces] = i;
			PARAMS::NActiveFaces++;
		}
		else {
			PARAMS::GridWGhost[i] = PARAMS::Grid[i];
			PARAMS::Border[i] = 0;
		}
	}
	PARAMS::FaceSize[0] = PARAMS::Grid[1] * PARAMS::Grid[2] * PARAMS::Grid[3];
	PARAMS::FaceSize[1] = PARAMS::Grid[0] * PARAMS::Grid[2] * PARAMS::Grid[3];
	PARAMS::FaceSize[2] = PARAMS::Grid[0] * PARAMS::Grid[1] * PARAMS::Grid[3];
	PARAMS::FaceSize[3] = PARAMS::Grid[0] * PARAMS::Grid[1] * PARAMS::Grid[2];
	PARAMS::FaceSizeG[0] = PARAMS::GridWGhost[1] * PARAMS::GridWGhost[2] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[1] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[2] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[2] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[1] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[3] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[1] * PARAMS::GridWGhost[2];

//MPI_Barrier(MPI_COMM_WORLD);
	/*printf("---->>>>>>>%d::%d:::::%d:%d:%d:%d:::::%d:%d:%d:%d\n",mynode(), numnodes(),PARAMS::GridWGhost[0],PARAMS::GridWGhost[1],\
		PARAMS::GridWGhost[2],PARAMS::GridWGhost[3],param_border(0),param_border(1),param_border(2),param_border(3));
	printf("---->>>>>>>%d::%d:::::%d:%d:%d:%d:::::%d:%d:%d:%d\n",mynode(), numnodes(),PARAMS::logical_coordinate[0],PARAMS::logical_coordinate[1],\
		PARAMS::logical_coordinate[2],PARAMS::logical_coordinate[3],nodes_per_dim(0),nodes_per_dim(1),nodes_per_dim(2),nodes_per_dim(3));*/
//MPI_Barrier(MPI_COMM_WORLD);
	int temp[NDIMS];
	for(int i = 0; i < NDIMS; i++) temp[i] = PARAMS::logical_coordinate[i];
	for(int fc = 0; fc < PARAMS::NActiveFaces; fc++){
		int i = PARAMS::FaceId[fc];
		temp[i] = (PARAMS::logical_coordinate[i] + 1) % nodes_per_dim(i);
		PARAMS::NodeIdRight[fc] = temp[0] + temp[1] * nodes_per_dim(0);
		PARAMS::NodeIdRight[fc] += temp[2] * nodes_per_dim(0) * nodes_per_dim(1);
		PARAMS::NodeIdRight[fc] += temp[3] * nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2);
		temp[i] = PARAMS::logical_coordinate[i];

		temp[i] = (PARAMS::logical_coordinate[i] - 1 + nodes_per_dim(i)) % nodes_per_dim(i);
		PARAMS::NodeIdLeft[fc] = temp[0] + temp[1] * nodes_per_dim(0);
		PARAMS::NodeIdLeft[fc] += temp[2] * nodes_per_dim(0) * nodes_per_dim(1);
		PARAMS::NodeIdLeft[fc] += temp[3] * nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2);
		temp[i] = PARAMS::logical_coordinate[i];

		//printf("##%d:::%d:%d:%d:%d\n",mynode(), fc,PARAMS::NodeIdLeft[fc],PARAMS::NodeIdRight[fc],i);
	}

	PARAMS::kstride = PARAMS::Grid[0] * PARAMS::Grid[1];
	PARAMS::tstride = 1;
	for(int i=0; i<NDIMS-1; i++) PARAMS::tstride *= PARAMS::Grid[i];

	PARAMS::Volume = 1;
	PARAMS::VolumeG = 1;
	for(int i = 0; i < NDIMS; i++){
		PARAMS::Volume *= PARAMS::Grid[i];
		PARAMS::VolumeG *= PARAMS::GridWGhost[i];
	}
	PARAMS::HalfVolume = PARAMS::Volume / 2;
	PARAMS::HalfVolumeG = PARAMS::VolumeG / 2;
	PARAMS::size = PARAMS::Volume  * NDIMS;


	if(verbose) PrintDetails();
	//Set Device parameters
	//Copy to GPU constant memory
	copyConstantsToGPU();
	int dev;
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	cudaGetDeviceProperties(&PARAMS::deviceProp, dev);
	PARAMS::GPUGridDimX = PARAMS::deviceProp.maxGridSize[0];

	 


	PARAMS::nthreadsPHB = dim3(64,1,1);
	PARAMS::nblocksPHB = GetBlockDim(PARAMS::nthreadsPHB.x, PARAMS::HalfVolume);
	PARAMS::nthreadsOVR = dim3(128,1,1);
	PARAMS::nblocksOVR = GetBlockDim(PARAMS::nthreadsOVR.x, PARAMS::HalfVolume);
	PARAMS::nthreadsINIT = dim3(128,1,1); 
	PARAMS::nblocksINIT = GetBlockDim(PARAMS::nthreadsINIT.x, PARAMS::Volume);
	PARAMS::nthreadsINITHALF = dim3(128,1,1); 
	PARAMS::nblocksINITHALF = GetBlockDim(PARAMS::nthreadsINITHALF.x, PARAMS::HalfVolume);
	PARAMS::nthreadsREU = dim3(128,1,1);
	PARAMS::nblocksREU = GetBlockDim(PARAMS::nthreadsREU.x, PARAMS::size);
	PARAMS::nthreadsPLAQ = dim3(64,1,1); //best 32^4 -> 64, next was for 128
	PARAMS::nblocksPLAQ =GetBlockDim(PARAMS::nthreadsPLAQ.x, PARAMS::Volume);


	//setparams = true;
	#ifdef MULTI_GPU
	FreeTempBuffersAndStreams();
	#endif
}



void SETPARAMS(bool _usetex, int latticedim[4], const int nodesperdim[4], \
    const int logical_coordinate[4], bool verbose){
	/*static bool setparams = false;
	if(setparams){
		COUT << "Local parameters already set... Nothing to do here..." << std::endl;
		return;
	}*/
	COUT << "Setting up lattice parameters..." << std::endl;

	PARAMS::UseTex = _usetex;
	PARAMS::NX = latticedim[0];
	PARAMS::NY = latticedim[1];
	PARAMS::NZ = latticedim[2];
	PARAMS::NT = latticedim[3];
	PARAMS::Grid[0] = latticedim[0] / nodesperdim[0];
	PARAMS::Grid[1] = latticedim[1] / nodesperdim[1];
	PARAMS::Grid[2] = latticedim[2] / nodesperdim[2];
	PARAMS::Grid[3] = latticedim[3] / nodesperdim[3];
	////////////////////////////////////////////
	for(int i = 0; i < 4; i++) PARAMS::logical_coordinate[i]= logical_coordinate[i];

	SetMPIParam_MILC(latticedim, logical_coordinate, nodesperdim);


	for(int i = 0; i < 4; i++) 
		if( ((PARAMS::Grid[i] / nodes_per_dim(i)) % 2) != 0){
			COUT << "GPU code does not support odd lattice dimensions." << std::endl;
			exit(1);
		}

	PARAMS::NActiveFaces = 0;
	for(int i = 0; i < 4; i++){
		if(nodes_per_dim(i) > 1 ) {
			PARAMS::Border[i] = RADIUS_BORDER;
			PARAMS::GridWGhost[i] = PARAMS::Grid[i] + 2 * PARAMS::Border[i];
			PARAMS::FaceId[PARAMS::NActiveFaces] = i;
			PARAMS::NActiveFaces++;
		}
		else {
			PARAMS::GridWGhost[i] = PARAMS::Grid[i];
			PARAMS::Border[i] = 0;
		}
	}
	PARAMS::FaceSize[0] = PARAMS::Grid[1] * PARAMS::Grid[2] * PARAMS::Grid[3];
	PARAMS::FaceSize[1] = PARAMS::Grid[0] * PARAMS::Grid[2] * PARAMS::Grid[3];
	PARAMS::FaceSize[2] = PARAMS::Grid[0] * PARAMS::Grid[1] * PARAMS::Grid[3];
	PARAMS::FaceSize[3] = PARAMS::Grid[0] * PARAMS::Grid[1] * PARAMS::Grid[2];
	PARAMS::FaceSizeG[0] = PARAMS::GridWGhost[1] * PARAMS::GridWGhost[2] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[1] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[2] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[2] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[1] * PARAMS::GridWGhost[3];
	PARAMS::FaceSizeG[3] = PARAMS::GridWGhost[0] * PARAMS::GridWGhost[1] * PARAMS::GridWGhost[2];


	int temp[4];
	for(int i = 0; i < 4; i++) temp[i] = PARAMS::logical_coordinate[i];
	for(int fc = 0; fc < PARAMS::NActiveFaces; fc++){
		int i = PARAMS::FaceId[fc];
		temp[i] = (PARAMS::logical_coordinate[i] + 1) % nodes_per_dim(i);
		PARAMS::NodeIdRight[fc] = temp[0] + temp[1] * nodes_per_dim(0);
		PARAMS::NodeIdRight[fc] += temp[2] * nodes_per_dim(0) * nodes_per_dim(1);
		PARAMS::NodeIdRight[fc] += temp[3] * nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2);
		temp[i] = PARAMS::logical_coordinate[i];

		temp[i] = (PARAMS::logical_coordinate[i] - 1 + nodes_per_dim(i)) % nodes_per_dim(i);
		PARAMS::NodeIdLeft[fc] = temp[0] + temp[1] * nodes_per_dim(0);
		PARAMS::NodeIdLeft[fc] += temp[2] * nodes_per_dim(0) * nodes_per_dim(1);
		PARAMS::NodeIdLeft[fc] += temp[3] * nodes_per_dim(0) * nodes_per_dim(1) * nodes_per_dim(2);
		temp[i] = PARAMS::logical_coordinate[i];

		//printf("##%d:::%d:%d:%d:%d\n",mynode(), fc,PARAMS::NodeIdLeft[fc],PARAMS::NodeIdRight[fc],i);
	}

	PARAMS::kstride = PARAMS::Grid[0] * PARAMS::Grid[1];
	PARAMS::tstride = PARAMS::kstride * PARAMS::Grid[2];    

	PARAMS::Volume = 1;
	PARAMS::VolumeG = 1;
	for(int i = 0; i < 4; i++){
		PARAMS::Volume *= PARAMS::Grid[i];
		PARAMS::VolumeG *= PARAMS::GridWGhost[i];
	}
	PARAMS::HalfVolume = PARAMS::Volume / 2;
	PARAMS::HalfVolumeG = PARAMS::VolumeG / 2;
	PARAMS::size = PARAMS::Volume  * 4;


	if(verbose) PrintDetails();
	//Set Device parameters
	//Copy to GPU constant memory
	copyConstantsToGPU();
	int dev;
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	cudaGetDeviceProperties(&PARAMS::deviceProp, dev);
	PARAMS::GPUGridDimX = PARAMS::deviceProp.maxGridSize[0];



	PARAMS::nthreadsPHB = dim3(64,1,1);
	PARAMS::nblocksPHB = GetBlockDim(PARAMS::nthreadsPHB.x, PARAMS::HalfVolume);
	PARAMS::nthreadsOVR = dim3(128,1,1);
	PARAMS::nblocksOVR = GetBlockDim(PARAMS::nthreadsOVR.x, PARAMS::HalfVolume);
	PARAMS::nthreadsINIT = dim3(128,1,1); 
	PARAMS::nblocksINIT = GetBlockDim(PARAMS::nthreadsINIT.x, PARAMS::Volume);
	PARAMS::nthreadsINITHALF = dim3(128,1,1); 
	PARAMS::nblocksINITHALF = GetBlockDim(PARAMS::nthreadsINITHALF.x, PARAMS::HalfVolume);
	PARAMS::nthreadsREU = dim3(128,1,1);
	PARAMS::nblocksREU = GetBlockDim(PARAMS::nthreadsREU.x, PARAMS::size);
	PARAMS::nthreadsPLAQ = dim3(64,1,1); //best 32^4 -> 64, next was for 128
	PARAMS::nblocksPLAQ =GetBlockDim(PARAMS::nthreadsPLAQ.x, PARAMS::Volume);


	//setparams = true;
	#ifdef MULTI_GPU
	FreeTempBuffersAndStreams();
	#endif
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace DEVPARAMS{
    /*! \brief If true uses texture memory instead of global memory */
    __constant__	bool	UseTex;
    /*! \brief inverse of the gauge coupling, beta = 2 Nc / g_0^2 */
    __constant__	double	Beta;
    /*! \brief inverse of the gauge coupling over Nc, betaOverNc = 2 / g_0^2 */
    __constant__ double   BetaOverNc;
    /*! \brief Size = Nx x Ny */
    __constant__	int	 	kstride;
    /*! \brief Size = Nx x Ny x Nz */
    __constant__	int	 	tstride;
    /*! \brief Size of the total gauge field, Nx x Ny x Nz x Nt x 4 */
    __constant__	int		size;

    __constant__	int	 	Grid[NDIMS];
    /*! \brief Size = Nx x Ny x Nz x Nt */
    __constant__	int	 	Volume;
    /*! \brief Size = Nx x Ny x Nz x Nt / 2 */
    __constant__	int	 	HalfVolume;
    __constant__	int	 	GridWGhost[NDIMS];
    __constant__	int	 	Border[NDIMS];
    __constant__	int	 	VolumeG;
    __constant__	int	 	HalfVolumeG;

    /*! \brief HYP smearing constant: alpha1 */
    __constant__	float	hypalpha1;
    /*! \brief HYP smearing constant: alpha2 */
    __constant__	float	hypalpha2;
    /*! \brief HYP smearing constant: alpha3 */
    __constant__	float	hypalpha3;

}

/*! \brief Setups the texture memory reading, if TexOn is false then uses global memory */
void UseTextureMemory(bool TexOn){
    PARAMS::UseTex = TexOn;
	memcpyToSymbol(DEVPARAMS::UseTex, &PARAMS::UseTex, bool);
}


void copyConstantsToGPU(){
	COUT << "Copying lattice constants to GPU Constant memory." << std::endl;
	memcpyToSymbol(DEVPARAMS::UseTex, &PARAMS::UseTex, bool);
	memcpyToSymbol(DEVPARAMS::Beta, &PARAMS::Beta, double);
	double _betaOverNc = PARAMS::Beta / (double) NCOLORS;
	memcpyToSymbol(DEVPARAMS::BetaOverNc, &_betaOverNc, double);
	memcpyToArraySymbol( DEVPARAMS::Grid, &PARAMS::Grid, int, NDIMS);
	memcpyToArraySymbol( DEVPARAMS::GridWGhost, &PARAMS::GridWGhost, int, NDIMS);
	memcpyToArraySymbol( DEVPARAMS::Border, &PARAMS::Border, int, NDIMS);
	memcpyToSymbol( DEVPARAMS::kstride, &PARAMS::kstride, int);
	memcpyToSymbol( DEVPARAMS::tstride, &PARAMS::tstride, int) ;
	memcpyToSymbol( DEVPARAMS::size, &PARAMS::size, int);
	memcpyToSymbol( DEVPARAMS::Volume, &PARAMS::Volume, int);
	memcpyToSymbol( DEVPARAMS::HalfVolume, &PARAMS::HalfVolume, int);
	memcpyToSymbol( DEVPARAMS::VolumeG, &PARAMS::VolumeG, int);
	memcpyToSymbol( DEVPARAMS::HalfVolumeG, &PARAMS::HalfVolumeG, int);
}



void copyHYPSmearConstants(float _alpha1, float _alpha2, float _alpha3){
  std::cout << "Copying HYP smearing constants to GPU Constant memory." << std::endl;
  memcpyToSymbol(DEVPARAMS::hypalpha1, &_alpha1, float);
  memcpyToSymbol(DEVPARAMS::hypalpha2, &_alpha2, float);
  memcpyToSymbol(DEVPARAMS::hypalpha3, &_alpha3, float);
}

}

