/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
 
#ifndef REDUCTION_H
#define REDUCTION_H

inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line );

extern "C"
bool isPow2(unsigned int x);

unsigned int nextPow2( unsigned int x );

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);



template <class T> 
T reduction(T *array_d, int size);


template <class T> 
T reduction(T *array_d, int size, const cudaStream_t &stream );

#endif
