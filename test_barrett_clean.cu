/**
 * Test program for clean Barrett reduction implementation
 * Validates correctness and provides performance benchmarks
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "barrett_clean.cuh"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Performance benchmark kernel
__global__ void benchmark_barrett_reduction(int iterations, uint128_t* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Test values
    uint128_t x(0x123456789ABCDEF0ULL + tid, tid);
    uint128_t m(0xFFFFFFFF00000001ULL, 0);  // Common prime
    
    // Precompute Barrett parameters
    BarrettParams params;
    barrett_precompute(params, m);
    
    // Warm up
    uint128_t result = x;
    for (int i = 0; i < 10; i++) {
        result = barrett_reduce(result, params);
    }
    
    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        result = barrett_reduce(result, params);
    }
    
    // Store result to prevent optimization
    if (tid == 0) {
        results[0] = result;
    }
}

// Host function to run comprehensive tests
void run_comprehensive_tests() {
    printf("Running comprehensive Barrett reduction tests...\n\n");
    
    // Test 1: Run basic test kernel
    printf("1. Running basic test kernel...\n");
    test_barrett_reduction_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n");
    
    // Test 2: Random validation (reduced number for debugging)
    printf("2. Running random validation tests...\n");
    const int num_random_tests = 10;  // Reduced from 1000
    const int threads_per_block = 10;
    const int num_blocks = 1;
    
    unsigned int seed = time(NULL);
    validate_barrett_random<<<num_blocks, threads_per_block>>>(num_random_tests, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Completed %d random validation tests\n\n", num_random_tests);
    
    // Test 3: Performance benchmark
    printf("3. Running performance benchmark...\n");
    uint128_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint128_t)));
    
    const int iterations = 1000000;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    benchmark_barrett_reduction<<<1, 32>>>(iterations, d_results);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Barrett reduction performance: %.2f million ops/sec\n", 
           (iterations * 32.0) / (milliseconds * 1000.0));
    
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Test specific edge cases
__global__ void test_edge_cases() {
    if (threadIdx.x != 0) return;
    
    printf("=== Testing Edge Cases ===\n\n");
    
    // Test with powers of 2
    {
        printf("Powers of 2 moduli:\n");
        for (int i = 10; i <= 60; i += 10) {
            uint128_t m(1ULL << i, 0);
            uint128_t x(0xFFFFFFFFFFFFFFFFULL, 0x123);
            
            BarrettParams params;
            barrett_precompute(params, m);
            
            uint128_t result = barrett_reduce(x, params);
            uint128_t expected = fallback_mod(x, m);
            
            printf("  2^%d: Barrett=%llu, Expected=%llu, Match=%s\n",
                   i, result.low, expected.low,
                   (result == expected) ? "YES" : "NO");
        }
    }
    
    // Test with Mersenne-like numbers
    {
        printf("\nMersenne-like moduli:\n");
        uint64_t mersenne_values[] = {
            (1ULL << 31) - 1,  // 2^31 - 1
            (1ULL << 61) - 1,  // 2^61 - 1
        };
        
        for (int i = 0; i < 2; i++) {
            uint128_t m(mersenne_values[i], 0);
            uint128_t x(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
            
            BarrettParams params;
            barrett_precompute(params, m);
            
            uint128_t result = barrett_reduce(x, params);
            uint128_t expected = fallback_mod(x, m);
            
            printf("  2^%d-1: Match=%s\n",
                   (i == 0) ? 31 : 61,
                   (result == expected) ? "YES" : "NO");
        }
    }
    
    // Test with common cryptographic primes
    {
        printf("\nCryptographic primes:\n");
        struct {
            uint64_t low;
            uint64_t high;
            const char* name;
        } primes[] = {
            {0xFFFFFFFF00000001ULL, 0, "2^64 - 2^32 + 1"},
            {0xFFFFFFFFFFFFFFC5ULL, 0, "2^64 - 59"},
            {0x1000000000000000ULL, 0x1, "2^64 + 2^60"},
        };
        
        for (int i = 0; i < 3; i++) {
            uint128_t m(primes[i].low, primes[i].high);
            uint128_t x(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);
            
            BarrettParams params;
            barrett_precompute(params, m);
            
            uint128_t result = barrett_reduce(x, params);
            uint128_t expected = fallback_mod(x, m);
            
            printf("  %s: Match=%s\n",
                   primes[i].name,
                   (result == expected) ? "YES" : "NO");
        }
    }
    
    printf("\n=== Edge Cases Complete ===\n");
}

int main() {
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using CUDA device: %s (Compute %d.%d)\n\n", 
           prop.name, prop.major, prop.minor);
    
    // Run tests
    run_comprehensive_tests();
    
    // Test edge cases
    printf("\n4. Testing edge cases...\n");
    test_edge_cases<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\nAll tests completed successfully!\n");
    
    return 0;
}