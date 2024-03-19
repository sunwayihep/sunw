
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>



#include <timer.h>
#include <cuda_common.h>
#include <device_load_save.h>
#include <constants.h>
#include <matrixsun.h>
#include <gaugearray.h>
#include <index.h>
#include <device_PHB_OVR.h>
#include <reunitlink.h>
#include <staple.h>
#include <comm_mpi.h>
#include <exchange.h>
#include <texture_host.h>

#include <sharedmemtypes.h>

#include <tune.h>
#include <launch_kernel.cuh>


#include <cudaAtomic.h>

#include <cub/cub.cuh>



using namespace std;


namespace CULQCD{







template<class Real>
struct ChromoFieldArg{
  complex *ploop;
  Real *field;
  Real *pl;
  complex *plaq;
  int radius;
  int nx;
  int ny;
};





template<int blockSize, bool UseTex, class Real, bool ppdagger, bool EvenRadius>
__global__ void kernel_ChromoField(ChromoFieldArg<Real> arg){
  typedef cub::BlockReduce<Real, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;  
        
	int id = INDEX1D();
  complex plup = complex::zero();
  if(id < DEVPARAMS::tstride)   plup = arg.ploop[id];
  Real value = 0.0;
  for(int dir2 = 0; dir2 < 3; dir2++){
    Real loop = 0.0;
    if(id < DEVPARAMS::tstride){  
      complex pldown = arg.ploop[Index_3D_Neig_NM(id, dir2, arg.radius)];
      if(ppdagger) loop = (plup * pldown.conj()).real() / (Real)(NCOLORS * NCOLORS);
      else loop = (plup * pldown).real() / (Real)(NCOLORS * NCOLORS);
    }
    
    int x[3];
    Index_3D_NM(id, x);
    x[dir2] = (x[dir2] + arg.radius / 2) % DEVPARAMS::Grid[dir2];
    

    for( int ix = 0; ix < arg.nx; ++ix )
    for( int iy = 0; iy < arg.ny; ++iy ) {
    
      Real field[6];
      for(int dd = 0; dd < 6; dd++) field[dd] = 0.0;

      	if(EvenRadius){
		  for(int dir1 = 0; dir1 < 3; dir1++){
		    if(dir1==dir2) continue;
		    int dir3 = 0;
		    for(int dir33 = 0; dir33 < 3; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;
		  
	  		  int s = Index_3D_Neig_NM(x, dir1, ix - arg.nx / 2, dir2, iy - arg.ny / 2);
		    
		    if(id < DEVPARAMS::tstride){
		      //Ex^2
		      //Real plaq = arg.plaq[s + dir1 * DEVPARAMS::tstride].real();
		      Real plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir1 * DEVPARAMS::tstride).real();
		      field[0] += plaq;
		      //Ey^2
		      //plaq = arg.plaq[s + dir2 * DEVPARAMS::tstride].real();
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir2 * DEVPARAMS::tstride).real();
		      field[1] += plaq;
		      //Ez^2
		      int s1 = Index_3D_Neig_NM(s, dir3, -1);
		      /*plaq = arg.plaq[s + dir3 * DEVPARAMS::tstride].real();
		      plaq += arg.plaq[s1 + dir3 * DEVPARAMS::tstride].real();*/
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir3 * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + dir3 * DEVPARAMS::tstride).real();
		      field[2] += plaq * 0.5;
		      //Bx^2
		      /*plaq = arg.plaq[s + (3 + dir1) * DEVPARAMS::tstride].real();
		      plaq += arg.plaq[s1 + (3 + dir1) * DEVPARAMS::tstride].real();*/
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir1) * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + (3 + dir1) * DEVPARAMS::tstride).real();
		      field[3] += plaq * 0.5;
		      //By^2
		      /*plaq = arg.plaq[s + (3 + dir2) * DEVPARAMS::tstride].real();
		      plaq += arg.plaq[s1 + (3 + dir2) * DEVPARAMS::tstride].real();*/
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir2) * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + (3 + dir2) * DEVPARAMS::tstride).real();
		      field[4] += plaq * 0.5;
		      //Bz^2
		      //plaq = arg.plaq[s + (3 + dir3) * DEVPARAMS::tstride].real();
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir3) * DEVPARAMS::tstride).real();
		      field[5] += plaq;
		    }
		  }
		  for(int dd = 0; dd < 6; dd++) field[dd] *= loop;
		  Real aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    aggregate[dd] = BlockReduce(temp_storage).Reduce(field[dd], Summ<Real>());
      		__syncthreads();
		  }
		    
		  int fieldoffset = arg.nx * arg.ny;
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + fieldoffset, aggregate[1]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + fieldoffset, aggregate[1]); 
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset, aggregate[3]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset, aggregate[3]); 
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
		  } 
	  }
	  else{
		  for(int dir1 = 0; dir1 < 3; dir1++){
			if(dir1==dir2) continue;
			int dir3 = 0;
			for(int dir33 = 0; dir33 < 3; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;
		  
	  		  int s = Index_3D_Neig_NM(x, dir1, ix - arg.nx / 2, dir2, iy - arg.ny / 2);
			
			if(id < DEVPARAMS::tstride){
		      //Ex^2
		      Real plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir1 * DEVPARAMS::tstride).real();
		      field[0] += plaq * 0.25;
		      //Ey^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir2 * DEVPARAMS::tstride).real();
		      field[1] += plaq;
		      //Ez^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir3 * DEVPARAMS::tstride).real();
		      int s1 = Index_3D_Neig_NM(s, dir3, -1);
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + dir3 * DEVPARAMS::tstride).real();
		      field[2] += plaq * 0.25;
		      //Bx^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir1) * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + (3 + dir1) * DEVPARAMS::tstride).real();
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir2) * DEVPARAMS::tstride).real(); 
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + (3 + dir2) * DEVPARAMS::tstride).real();
		      field[4] += plaq * 0.125;
		      //Bz^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir3) * DEVPARAMS::tstride).real();
		      field[5] += plaq * 0.5;	    
		    }
		  }
		  for(int dd = 0; dd < 6; dd++) field[dd] *= loop;
		  Real aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    aggregate[dd] = BlockReduce(temp_storage).Reduce(field[dd], Summ<Real>());
      		__syncthreads();
		  }	    
		  int fieldoffset = arg.nx * arg.ny;
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1, aggregate[0]); 
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1, aggregate[0]);
			id1 =  ((ix + 1) % arg.nx) + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + fieldoffset, aggregate[1]);
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset, aggregate[2]);
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 2 * fieldoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset, aggregate[3]);
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
			id1 =  ix + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
			id1 =  ((ix + 1) % arg.nx) + ((iy - 1 + arg.ny) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
		  } 
	  }
      __syncthreads();
    }
    value += loop;
  }
	Real pl = 0.;
	pl = BlockReduce(temp_storage).Reduce(value, Summ<Real>());
	__syncthreads();
	if (threadIdx.x == 0) CudaAtomicAdd(arg.pl, pl);
}




template<int blockSize, bool UseTex, class Real, bool ppdagger, bool EvenRadius>
__global__ void kernel_ChromoFieldMidFluxTube(ChromoFieldArg<Real> arg){
  typedef cub::BlockReduce<Real, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;  
        
	int id = INDEX1D();
  complex plup = complex::zero();
  if(id < DEVPARAMS::tstride)   plup = arg.ploop[id];
  Real value = 0.0;
  for(int dir2 = 0; dir2 < 3; dir2++){
    Real loop = 0.0;
    if(id < DEVPARAMS::tstride){  
      complex pldown = arg.ploop[Index_3D_Neig_NM(id, dir2, arg.radius)];
      if(ppdagger) loop = (plup * pldown.conj()).real() / (Real)(NCOLORS * NCOLORS);
      else loop = (plup * pldown).real() / (Real)(NCOLORS * NCOLORS);
    }
    
    int x[3];
    Index_3D_NM(id, x);
    x[dir2] = (x[dir2] + arg.radius / 2) % DEVPARAMS::Grid[dir2];
    

    for( int ix = 0; ix < arg.nx; ++ix )
    for( int iy = 0; iy < arg.ny; ++iy ) {
    
      Real field[6];
      for(int dd = 0; dd < 6; dd++) field[dd] = 0.0;

      	if(EvenRadius){      
		  for(int dir1 = 0; dir1 < 3; dir1++){
		    if(dir1==dir2) continue;
		    int dir3 = 0;
		    for(int dir33 = 0; dir33 < 3; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;
		    
	  		  int s = Index_3D_Neig_NM(x, dir1, ix - arg.nx / 2, dir3, iy - arg.ny / 2);
		    
		    if(id < DEVPARAMS::tstride){
		      //Ex^2
		      Real plaq = ELEM_LOAD<UseTex, Real>(arg.plaq,s + dir1 * DEVPARAMS::tstride).real();
		      field[0] += plaq;
		      //Ey^2
		      int s1 = Index_3D_Neig_NM(s, dir2, -1);
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq,s + dir2 * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq,s1 + dir2 * DEVPARAMS::tstride).real();
		      field[1] += plaq * 0.5;
		      //Ez^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq,s + dir3 * DEVPARAMS::tstride).real();
		      field[2] += plaq;
		      //Bx^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq,s + (3 + dir1) * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq,s1 + (3 + dir1) * DEVPARAMS::tstride).real();
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq,s + (3 + dir2) * DEVPARAMS::tstride).real();
		      field[4] += plaq;
		      //Bz^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq,s + (3 + dir3) * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq,s1 + (3 + dir3) * DEVPARAMS::tstride).real();
		      field[5] += plaq * 0.5;
		      /*//Ex^2
		      Real plaq = arg.plaq[s + dir1 * DEVPARAMS::tstride].real();
		      field[0] += loop * plaq;
		      //Ey^2
		      int s1 = Index_3D_Neig_NM(s, dir2, -1);
		      plaq = arg.plaq[s + dir2 * DEVPARAMS::tstride].real();
		      plaq += arg.plaq[s1 + dir2 * DEVPARAMS::tstride].real();
		      field[1] += loop * plaq * 0.5;
		      //Ez^2
		      plaq = arg.plaq[s + dir3 * DEVPARAMS::tstride].real();
		      field[2] += loop * plaq;
		      //Bx^2
		      plaq = arg.plaq[s + (3 + dir1) * DEVPARAMS::tstride].real();
		      plaq += arg.plaq[s1 + (3 + dir1) * DEVPARAMS::tstride].real();
		      field[3] += loop * plaq * 0.5;
		      //By^2
		      plaq = arg.plaq[s + (3 + dir2) * DEVPARAMS::tstride].real();
		      field[4] += loop * plaq;
		      //Bz^2
		      plaq = arg.plaq[s + (3 + dir3) * DEVPARAMS::tstride].real();
		      plaq += arg.plaq[s1 + (3 + dir3) * DEVPARAMS::tstride].real();
		      field[5] += loop * plaq * 0.5;*/
		    }
		  }
		  for(int dd = 0; dd < 6; dd++) field[dd] *= loop;
		  Real aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    aggregate[dd] = BlockReduce(temp_storage).Reduce(field[dd], Summ<Real>());
      		__syncthreads();
		  }
		    

		  int fieldoffset = arg.nx * arg.ny;
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1, aggregate[0]);
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + fieldoffset, aggregate[1]);
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset, aggregate[2]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 2 * fieldoffset, aggregate[2]); 
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset, aggregate[3]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset, aggregate[3]); 
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]);
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
		  } 
	  }
	  else{
		  for(int dir1 = 0; dir1 < 3; dir1++){
			if(dir1==dir2) continue;
			int dir3 = 0;
			for(int dir33 = 0; dir33 < 3; dir33++) if(dir1 != dir33 && dir2 != dir33) dir3 = dir33;
		  
	  		  int s = Index_3D_Neig_NM(x, dir1, ix - arg.nx / 2, dir3, iy - arg.ny / 2);
			
			if(id < DEVPARAMS::tstride){
		      //Ex^2
		      Real plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir1 * DEVPARAMS::tstride).real();
		      int s1 = Index_3D_Neig_NM(s, dir2, 1);
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + dir1 * DEVPARAMS::tstride).real();
		      field[0] += plaq * 0.25;
		      //Ey^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir2 * DEVPARAMS::tstride).real();
		      field[1] += plaq;
		      //Ez^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + dir3 * DEVPARAMS::tstride).real();
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + dir3 * DEVPARAMS::tstride).real();
		      field[2] += plaq * 0.25;
		      //Bx^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir1) * DEVPARAMS::tstride).real();
		      field[3] += plaq * 0.5;
		      //By^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir2) * DEVPARAMS::tstride).real(); 
		      plaq += ELEM_LOAD<UseTex, Real>(arg.plaq, s1 + (3 + dir2) * DEVPARAMS::tstride).real(); 
		      field[4] += plaq * 0.125;
		      //Bz^2
		      plaq = ELEM_LOAD<UseTex, Real>(arg.plaq, s + (3 + dir3) * DEVPARAMS::tstride).real();
		      field[5] += plaq * 0.5;
		    }
		  }
		  for(int dd = 0; dd < 6; dd++) field[dd] *= loop;
		  Real aggregate[6];
		  for(int dd = 0; dd < 6; dd++){
		    aggregate[dd] = BlockReduce(temp_storage).Reduce(field[dd], Summ<Real>());
      		__syncthreads();
		  }
		    
		  int fieldoffset = arg.nx * arg.ny;
		  if (threadIdx.x == 0){
		  //accum Ex^2
		    int id0 = ix + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id0, aggregate[0]);
		    int id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1, aggregate[0]); 
		  //accum Ey^2
		    CudaAtomicAdd(arg.field + id0 + fieldoffset, aggregate[1]);
		  //accum Ez^2
		    CudaAtomicAdd(arg.field + id0 + 2 * fieldoffset, aggregate[2]);
			id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 2 * fieldoffset, aggregate[2]);
		  //accum Bx^2
		    CudaAtomicAdd(arg.field + id0 + 3 * fieldoffset, aggregate[3]);
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 3 * fieldoffset, aggregate[3]); 
		  //accum By^2
		    CudaAtomicAdd(arg.field + id0 + 4 * fieldoffset, aggregate[4]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]); 
		    id1 =  ix + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]); 
		    id1 =  ((ix + 1) % arg.nx) + ((iy + 1) % arg.ny) * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 4 * fieldoffset, aggregate[4]); 
		  //accum Bz^2
		    CudaAtomicAdd(arg.field + id0 + 5 * fieldoffset, aggregate[5]);
		    id1 =  ((ix + 1) % arg.nx) + iy * arg.nx;
		    CudaAtomicAdd(arg.field + id1 + 5 * fieldoffset, aggregate[5]);
		  } 
	  }
      __syncthreads();
    }
    value += loop;
  }
	Real pl = 0.;
	pl = BlockReduce(temp_storage).Reduce(value, Summ<Real>());
	__syncthreads();
	if (threadIdx.x == 0) CudaAtomicAdd(arg.pl, pl);

}





	

template <bool UseTex, class Real, bool chargeplane, bool ppdagger, bool EvenRadius> 
class ChromoField: Tunable{
private:
   ChromoFieldArg<Real> arg;
   Real *chromofield;
   Real *pl;
   int size;
   double timesec;
#ifdef TIMMINGS
    Timer ChromoFieldtime;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  CUDA_SAFE_CALL(cudaMemset(arg.field, 0, 6 * arg.nx * arg.ny * sizeof(Real)));
  CUDA_SAFE_CALL(cudaMemset(arg.pl, 0, sizeof(Real)));
  if(chargeplane){ LAUNCH_KERNEL(kernel_ChromoField, tp, stream, arg, UseTex, Real, ppdagger, EvenRadius);}  
  else {LAUNCH_KERNEL(kernel_ChromoFieldMidFluxTube, tp, stream, arg, UseTex, Real, ppdagger, EvenRadius);  }
}

public:
   ChromoField(complex *ploop, complex *plaqfield, Real *chromofield, Real *pl, int radius, int nx, int ny):chromofield(chromofield), pl(pl){
	size = 1;
	for(int i=0;i<3;i++){
		size *= PARAMS::Grid[i];
	} 
	timesec = 0.0;
	arg.ploop = ploop;
	arg.plaq = plaqfield;
	arg.field = (Real *)dev_malloc( 6 * nx * ny * sizeof(Real));
	arg.pl = (Real *)dev_malloc( sizeof(Real));
	arg.radius = radius;
	arg.nx = nx;
	arg.ny = ny;
  
}
   ~ChromoField(){dev_free(arg.field);dev_free(arg.pl);};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    ChromoFieldtime.start();
#endif
    apply(stream);
    CUDA_SAFE_DEVICE_SYNC();
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(pl, arg.pl, sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(chromofield, arg.field, 6 * arg.nx * arg.ny * sizeof(Real), cudaMemcpyDeviceToHost));
    //normalize!!!!!!
	if(EvenRadius){
		for(int r = 0; r < 6 * arg.nx * arg.ny; r++)
		  chromofield[r] /= (Real)(12 * size);
		if(chargeplane){
		  for(int r = 2 * arg.nx * arg.ny; r < 3 * arg.nx * arg.ny; r++) //Ez^2
		    chromofield[r] *= 2.0; 
		  for(int r = 5 * arg.nx * arg.ny; r < 6 * arg.nx * arg.ny; r++) //Bz^2
		    chromofield[r] *= 0.5; 
		}
		else{
		  for(int r = arg.nx * arg.ny; r < 2 * arg.nx * arg.ny; r++) //Ey^2
		    chromofield[r] *= 2.0; 
		  for(int r = 4 * arg.nx * arg.ny; r < 5 * arg.nx * arg.ny; r++) //By^2
		    chromofield[r] *= 0.5; 
		}
	}
	else{
		for(int r = 0; r < 6 * arg.nx * arg.ny; r++)
		  chromofield[r] /= (Real)(6 * size);
	}
#ifdef TIMMINGS
	CUDA_SAFE_DEVICE_SYNC( );
    ChromoFieldtime.stop();
    timesec = ChromoFieldtime.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const {return 0;}
   long long bytes() const{return 0;}
   double time(){	return timesec;}
   void stat(){	COUT << "ChromoField:  " <<  time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
   void printValue(){}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size << ",prec="  << sizeof(Real);
    string tmp = "None";
    return TuneKey(vol.str().c_str(), typeid(*this).name(), tmp.c_str(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() {  }

};






template<bool UseTex, class Real, bool chargeplane, bool ppdagger, bool EvenRadius>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field, Real *pl, int radius, int nx, int ny){
  Timer mtime;
  mtime.start(); 
  ChromoField<UseTex, Real, chargeplane, ppdagger, EvenRadius> cfield(ploop, plaqfield, field, pl, radius, nx, ny);
  cfield.Run();
  cfield.stat();
  cfield.printValue();
  CUDA_SAFE_DEVICE_SYNC( );
  mtime.stop();
  COUT << "Time ChromoField:  " <<  mtime.getElapsedTimeInSec() << " s"  << endl;
}



template<bool UseTex, class Real, bool chargeplane, bool ppdagger>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field, Real *pl, 
                      int radius, int nx, int ny){
  if(radius % 2 == 0) CalcChromoField<UseTex, Real, chargeplane, ppdagger, true>(ploop, plaqfield, field, pl, radius, nx, ny);
  else CalcChromoField<UseTex, Real, chargeplane, ppdagger, false>(ploop, plaqfield, field, pl, radius, nx, ny);
}

template<bool UseTex, class Real, bool chargeplane>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field, Real *pl, 
                      int radius, int nx, int ny, bool ppdagger){
  if(ppdagger) CalcChromoField<UseTex, Real, chargeplane, true>(ploop, plaqfield, field, pl, radius, nx, ny);
  else CalcChromoField<UseTex, Real, chargeplane, false>(ploop, plaqfield, field, pl, radius, nx, ny);
}
template<bool UseTex, class Real>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field, Real *pl, 
                      int radius, int nx, int ny, bool chargeplane, bool ppdagger){
  if(chargeplane) CalcChromoField<UseTex, Real, true>(ploop, plaqfield, field, pl, radius, nx, ny, ppdagger);
  else CalcChromoField<UseTex, Real, false>(ploop, plaqfield, field, pl, radius, nx, ny, ppdagger);
}
template<class Real>
void CalcChromoField(complex *ploop, complex *plaqfield, Real *field, Real *pl, 
                      int radius, int nx, int ny, bool ppdagger, bool chargeplane){ 
  if(PARAMS::UseTex){
    GAUGE_TEXTURE(plaqfield, true);
    CalcChromoField<true, Real>(ploop, plaqfield, field, pl, radius, nx, ny, chargeplane, ppdagger);
  }
  else CalcChromoField<false, Real>(ploop, plaqfield, field, pl, radius, nx, ny, chargeplane, ppdagger);
}
template void CalcChromoField<float>(complexs *ploop, complexs *plaqfield, float *field, float *pl, int radius, int nx, int ny, bool ppdagger, bool chargeplane);
template void CalcChromoField<double>(complexd *ploop, complexd *plaqfield, double *field, double *pl, int radius, int nx, int ny, bool ppdagger, bool chargeplane);











}
