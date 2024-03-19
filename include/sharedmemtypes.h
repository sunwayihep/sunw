#ifndef SHARED_MEMORY_TYPES_H
#define SHARED_MEMORY_TYPES_H


#include <complex.h>
#include <matrixsun.h>



namespace CULQCD{

template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
        {
            extern __shared__ int __ssmem[];
            return (T*)__ssmem;
        }

    __device__ inline operator const T*() const
        {
            extern __shared__ int __ssmem[];
            return (T*)__ssmem;
        }
};
template<>
struct SharedMemory<complexs>
{
    __device__ inline operator       complexs*()
    {
        extern __shared__ complexs __ssmem_s[];
        return (complexs*)__ssmem_s;
    }

    __device__ inline operator const complexs*() const
    {
        extern __shared__ complexs __ssmem_s[];
        return (complexs*)__ssmem_s;
    }
};
template<>
struct SharedMemory<complexd>
{
    __device__ inline operator       complexd*()
    {
        extern __shared__ complexd __ssmem_d[];
        return (complexd*)__ssmem_d;
    }

    __device__ inline operator const complexd*() const
    {
        extern __shared__ complexd __ssmem_d[];
        return (complexd*)__ssmem_d;
    }
};

template<>
struct SharedMemory<msuns>
{
    __device__ inline operator       msuns*()
    {
        extern __shared__ msuns __smem_s[];
        return (msuns*)__smem_s;
    }

    __device__ inline operator const msuns*() const
    {
        extern __shared__ msuns __smem_s[];
        return (msuns*)__smem_s;
    }
};
template<>
struct SharedMemory<msund>
{
    __device__ inline operator       msund*()
    {
        extern __shared__ msund __smem_d[];
        return (msund*)__smem_d;
    }

    __device__ inline operator const msund*() const
    {
        extern __shared__ msund __smem_d[];
        return (msund*)__smem_d;
    }
};

}

#endif // #ifndef SHARED_MEMORY_TYPES_H
