/**
 * Test specific Barrett reduction case
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "barrett_clean.cuh"

__global__ void test_specific_case() {
    if (threadIdx.x != 0) return;
    
    // Test 3 from the main test - Full 128-bit modulus
    uint128_t x(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);  // 2^128 - 1
    uint128_t m(0x123456789ABCDEF0ULL, 0x1);  // Large 128-bit prime-like number
    
    printf("Testing specific case:\n");
    printf("x = 0x%llx:%llx\n", x.high, x.low);
    printf("m = 0x%llx:%llx\n", m.high, m.low);
    
    // Test bit_length function
    int x_bits = bit_length_128(x);
    int m_bits = bit_length_128(m);
    printf("bit_length(x) = %d\n", x_bits);
    printf("bit_length(m) = %d\n", m_bits);
    
    // Test fallback_mod
    printf("\nTesting fallback_mod...\n");
    uint128_t result = fallback_mod(x, m);
    printf("fallback_mod result = 0x%llx:%llx\n", result.high, result.low);
    
    // Test Barrett parameters
    printf("\nTesting Barrett parameters...\n");
    BarrettParams params;
    barrett_precompute(params, m);
    printf("mu = 0x%llx:%llx\n", params.mu.high, params.mu.low);
    printf("k = %d\n", params.k);
    
    // Test Barrett reduction
    printf("\nTesting barrett_reduce...\n");
    uint128_t barrett_result = barrett_reduce(x, params);
    printf("Barrett result = 0x%llx:%llx\n", barrett_result.high, barrett_result.low);
    
    printf("\nTest complete.\n");
}

int main() {
    test_specific_case<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}