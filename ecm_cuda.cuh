/**
 * Elliptic Curve Method (ECM) Header
 * Interface for GPU-accelerated ECM factorization
 */

#ifndef ECM_CUDA_CUH
#define ECM_CUDA_CUH

#include <cuda_runtime.h>
#include "uint128_improved.cuh"

// ECM function declarations
extern "C" bool ecm_factor(uint128_t n, uint128_t& factor, int max_curves = 1000);
__device__ __host__ bool ecm_is_suitable(uint128_t n);

// ECM configuration constants
#define ECM_DEFAULT_MAX_CURVES 1000
#define ECM_CURVES_PER_BATCH 64

#endif // ECM_CUDA_CUH