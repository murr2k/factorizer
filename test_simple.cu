/**
 * Simple test to verify basic functionality
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "uint128_improved.cuh"
#include "barrett_reduction.cuh"

__global__ void test_basic() {
    if (threadIdx.x == 0) {
        printf("Basic test starting...\n");
        
        // Test 1: uint128_t creation
        uint128_t a(90595490423ULL, 0);
        printf("Created number: %llu\n", a.low);
        
        // Test 2: Factors
        uint128_t f1(428759, 0);
        uint128_t f2(211297, 0);
        
        // Test 3: Multiplication
        uint256_t product = multiply_128_128(f1, f2);
        printf("Product: %llu (expected: 90595490423)\n", product.word[0]);
        
        // Test 4: GCD
        uint128_t g = gcd_128(a, f1);
        printf("GCD: %llu (expected: 428759)\n", g.low);
        
        printf("Basic test complete!\n");
    }
}

int main() {
    printf("Running simple test...\n");
    
    // Check device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    // Run test
    test_basic<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}