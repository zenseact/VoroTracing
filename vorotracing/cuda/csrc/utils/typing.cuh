#pragma once

#include <string>

#include <cuda_fp16.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

namespace vorotracing
{

template <typename T> __device__ __host__ inline float to_float(T val) { return (float)val; }

template <> __device__ __host__ inline float to_float(__half val) { return __half2float(val); }

template <typename T> __device__ __host__ inline T from_float(float val) { return (T)val; }

template <> __device__ __host__ inline __half from_float(float val) { return __float2half(val); }

template <typename T> __device__ __host__ inline void swap(T &a, T &b)
{
    typename std::decay<T>::type tmp = a;
    a = b;
    b = tmp;
}

/// @brief Compute the base-2 logarithm of an integer
inline __host__ __device__ uint32_t log2(uint32_t x)
{
#if defined(__CUDA_ARCH__)
    return (x > 0) ? 31 - __clz(x) : 0;
#else
    uint32_t result = 0;
    while (x >>= 1)
    {
        result++;
    }
    return result;
#endif
}

/// @brief Compute the smallest power of 2 greater than or equal to x
inline __host__ __device__ uint32_t pow2_round_up(uint32_t x) { return (x > 1) ? 1 << (log2(x - 1) + 1) : 1; }

} // namespace vorotracing