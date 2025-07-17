/**
 * Simple test for Barrett reduction debugging
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "barrett_clean.cuh"

__global__ void simple_barrett_test() {
    if (threadIdx.x != 0) return;
    
    printf("=== Simple Barrett Test ===\n");
    
    // Test 1: Very simple case
    uint128_t x(100, 0);
    uint128_t m(7, 0);
    
    BarrettParams params;
    params.modulus = m;
    params.k = 128;
    
    printf("Computing mu for modulus %llu...\n", m.low);
    params.mu = compute_barrett_mu(m, 128);
    printf("mu = 0x%llx:%llx\n", params.mu.high, params.mu.low);
    
    uint128_t result = barrett_reduce(x, params);
    uint128_t expected = uint128_t(100 % 7, 0);
    
    printf("100 mod 7 = %llu (expected: %llu)\n", result.low, expected.low);
    printf("Match: %s\n", (result == expected) ? "YES" : "NO");
}

int main() {
    simple_barrett_test<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    return 0;
}