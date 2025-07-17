/**
 * Simple test for v2.1.0 features
 * Tests individual components without full integration
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>

#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include "montgomery_reduction.cuh"

// Simple test kernel for Barrett reduction
__global__ void test_barrett_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== Testing Barrett Reduction v2 ===\n");
        
        // Test case: 12345678901234567890 mod 1000000007
        uint128_t a(0xAB54A98CEB1F0AD2ULL, 0x0ULL);
        uint128_t n(1000000007, 0);
        
        Barrett128_v2 barrett;
        barrett.n = n;
        barrett.precompute();
        
        uint128_t result = barrett.reduce_128(a);
        
        printf("Barrett test: %llx mod %llu = %llu\n", 
               a.low, n.low, result.low);
        printf("Expected: 652337934\n");
        printf("Result: %s\n", (result.low == 652337934) ? "PASSED" : "FAILED");
    }
}

// Simple test kernel for Montgomery reduction
__global__ void test_montgomery_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Testing Montgomery Reduction ===\n");
        
        // Test with odd modulus
        uint128_t n(1000000007, 0);
        
        Montgomery128 mont;
        mont.n = n;
        mont.precompute();
        
        // Test multiplication: 12345 * 67890 mod 1000000007
        uint128_t a(12345, 0);
        uint128_t b(67890, 0);
        uint128_t expected(838102050, 0);
        
        uint128_t a_mont = to_montgomery(a, mont);
        uint128_t b_mont = to_montgomery(b, mont);
        uint128_t result_mont = montgomery_multiply(a_mont, b_mont, mont);
        uint128_t result = from_montgomery(result_mont, mont);
        
        printf("Montgomery test: %llu * %llu mod %llu = %llu\n",
               a.low, b.low, n.low, result.low);
        printf("Expected: %llu\n", expected.low);
        printf("Result: %s\n", (result == expected) ? "PASSED" : "FAILED");
    }
}

// Performance comparison kernel
__global__ void performance_test_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Performance Comparison ===\n");
        
        uint128_t n(0xFFFFFFFFFFFFFFC5ULL, 0x7FFFFFFFFFFFFFFFULL);
        uint128_t a(0x123456789ABCDEFULL, 0x1ULL);
        uint128_t b(0xFEDCBA9876543210ULL, 0x2ULL);
        
        // Test Barrett v2
        Barrett128_v2 barrett;
        barrett.n = n;
        barrett.precompute();
        
        clock_t start = clock();
        uint128_t result = a;
        for (int i = 0; i < 1000; i++) {
            result = modmul_barrett_v2(result, b, barrett);
        }
        clock_t end = clock();
        
        double barrett_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        printf("Barrett v2 (1000 modmuls): %.3f ms\n", barrett_time);
        
        // Test Montgomery (only for odd modulus)
        if (n.low & 1) {
            Montgomery128 mont;
            mont.n = n;
            mont.precompute();
            
            uint128_t a_mont = to_montgomery(a, mont);
            uint128_t b_mont = to_montgomery(b, mont);
            
            start = clock();
            result = a_mont;
            for (int i = 0; i < 1000; i++) {
                result = montgomery_multiply(result, b_mont, mont);
            }
            result = from_montgomery(result, mont);
            end = clock();
            
            double mont_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
            printf("Montgomery (1000 modmuls): %.3f ms\n", mont_time);
            printf("Montgomery speedup: %.2fx\n", barrett_time / mont_time);
        }
    }
}

int main() {
    printf("Simple v2.1.0 Feature Test\n");
    printf("==========================\n\n");
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n\n", prop.name);
    
    // Run tests
    test_barrett_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    test_montgomery_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    performance_test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\nCUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("\nAll tests completed!\n");
    return 0;
}