
#ifndef CUFFT_PLANS_H
#define CUFFT_PLANS_H


#include <cufft.h>
#include <cufftXt.h>
#include <cuda_common.h>



namespace CULQCD{

inline void ApplyFFT(cufftHandle &plan, complexs *data_in, complexs *data_out, int direction){

    CUFFT_SAFE_CALL(cufftExecC2C(plan, (cufftComplex *)data_in, (cufftComplex *)data_out, direction));
}

inline void ApplyFFT(cufftHandle &plan, complexd *data_in, complexd *data_out, int direction){

    CUFFT_SAFE_CALL(cufftExecZ2Z(plan, (cufftDoubleComplex *)data_in, (cufftDoubleComplex *)data_out, direction));
}



inline void SetPlanFFTMany( cufftHandle &plan, int4 size, int dim, complexs *data){
    switch(dim){
    case 1:     
    {int n[1] = {size.w};
            cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.x * size.y * size.z);
    }
    break;
    case 3:
    {int n[3] = {size.x, size.y, size.z};
        cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.w);
    }
    break;
    }
    //printf("Created %dD FFT Plan in Single Precision\n", dim);
}

inline void SetPlanFFTMany( cufftHandle &plan, int4 size, int dim, complexd *data){
    switch(dim){
    case 1:     
    {int n[1] = {size.w};
            cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.x * size.y * size.z);
    }
    break;
    case 3:
    {int n[3] = {size.x, size.y, size.z};
        cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.w);
    }
    break;
    }
    //printf("Created %dD FFT Plan in Double Precision\n", dim);
}


inline void SetPlanFFT2DMany( cufftHandle &plan, int4 size, int dim, complexs *data){
    switch(dim){
    case 0:     
    {int n[2] = {size.w, size.z};
            cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.x * size.y);
    }
    break;
    case 1:
    //{int n[2] = {size.x, size.y};
    {int n[2] = {size.y, size.x};
        cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.z * size.w);
    }
    break;
    }
    //printf("Created 2D FFT Plan in Single Precision\n");
}

inline void SetPlanFFT2DMany( cufftHandle &plan, int4 size, int dim, complexd *data){
    switch(dim){
    case 0:     
    {int n[2] = {size.w, size.z};
            cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.x * size.y);
    }
    break;
    case 1:
    {int n[2] = {size.y, size.x};
    //{int n[2] = {size.x, size.y};
        cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.z * size.w);
    }
    break;
    }
    //printf("Created 2D FFT Plan in Double Precision\n");
}
}

#endif

