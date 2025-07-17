/**
 * Debug test for Barrett reduction
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "barrett_clean.cuh"

__global__ void debug_barrett_test() {
    if (threadIdx.x != 0) return;
    
    printf("=== Barrett Debug Test ===\n");
    
    // Test 1: Small 64-bit modulus
    {
        uint128_t x(0x123456789ABCDEF0ULL, 0);
        uint128_t m(1000000007, 0);
        
        printf("\nTest 1: Small modulus\n");
        printf("x = 0x%llx\n", x.low);
        printf("m = %llu\n", m.low);
        
        BarrettParams params;
        barrett_precompute(params, m);
        
        printf("mu computed = 0x%llx:%llx\n", params.mu.high, params.mu.low);
        
        uint128_t result = barrett_reduce(x, params);
        printf("Barrett result = %llu\n", result.low);
        
        // Compare with direct modulo
        uint64_t expected = x.low % m.low;
        printf("Expected = %llu\n", expected);
        printf("Match: %s\n\n", (result.low == expected) ? "YES" : "NO");
    }
    
    // Test 2: Large 128-bit modulus (but simple)
    {
        uint128_t x(0xFFFFFFFFFFFFFFFFULL, 0x1);
        uint128_t m(0x1, 0x1);  // 2^64 + 1
        
        printf("Test 2: 128-bit modulus\n");
        printf("x = 0x%llx:%llx\n", x.high, x.low);
        printf("m = 0x%llx:%llx\n", m.high, m.low);
        
        printf("Computing mu...\n");
        BarrettParams params;
        params.modulus = m;
        params.k = 128;
        params.mu = compute_barrett_mu(m, 128);
        
        printf("mu = 0x%llx:%llx\n", params.mu.high, params.mu.low);
        printf("Done computing mu\n");
    }
}

int main() {
    debug_barrett_test<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}