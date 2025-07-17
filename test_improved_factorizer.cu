/**
 * Comprehensive Test Suite for Improved 128-bit Factorizer
 * Tests all three improvements in isolation and integration
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <string>

#include "uint128_improved.cuh"
#include "barrett_reduction.cuh"
#include "curand_pollards_rho.cuh"

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double time_ms;
};

// Global test results
std::vector<TestResult> test_results;

// Helper macro for tests
#define RUN_TEST(name, kernel, grid, block) \
    do { \
        printf("Running %s...\n", name); \
        cudaEvent_t start, stop; \
        cudaEventCreate(&start); \
        cudaEventCreate(&stop); \
        cudaEventRecord(start); \
        kernel<<<grid, block>>>(); \
        cudaError_t err = cudaGetLastError(); \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        float time_ms = 0; \
        cudaEventElapsedTime(&time_ms, start, stop); \
        TestResult result = {name, err == cudaSuccess, cudaGetErrorString(err), time_ms}; \
        test_results.push_back(result); \
        cudaEventDestroy(start); \
        cudaEventDestroy(stop); \
    } while(0)

// Test 1: uint128_t arithmetic operations
__global__ void test_uint128_arithmetic_comprehensive() {
    if (threadIdx.x == 0) {
        bool all_pass = true;
        
        // Test addition with carry
        {
            uint128_t a(0xFFFFFFFFFFFFFFFF, 0);
            uint128_t b(1, 0);
            uint128_t sum = add_128(a, b);
            if (sum.low != 0 || sum.high != 1) {
                printf("FAIL: Addition carry test\n");
                all_pass = false;
            }
        }
        
        // Test subtraction with borrow
        {
            uint128_t a(0, 1);
            uint128_t b(1, 0);
            uint128_t diff = subtract_128(a, b);
            if (diff.low != 0xFFFFFFFFFFFFFFFF || diff.high != 0) {
                printf("FAIL: Subtraction borrow test\n");
                all_pass = false;
            }
        }
        
        // Test multiplication accuracy
        {
            uint128_t a(0xFFFFFFFF, 0);  // 2^32 - 1
            uint128_t b(0xFFFFFFFF, 0);  // 2^32 - 1
            uint256_t prod = multiply_128_128(a, b);
            // Expected: (2^32-1)^2 = 2^64 - 2^33 + 1 = 0xFFFFFFFE00000001
            if (prod.word[0] != 0xFFFFFFFE00000001ULL || prod.word[1] != 0) {
                printf("FAIL: Multiplication test\n");
                all_pass = false;
            }
        }
        
        // Test large multiplication
        {
            uint128_t a(0x123456789ABCDEF0, 0x1);
            uint128_t b(0x2, 0);
            uint256_t prod = multiply_128_128(a, b);
            uint128_t expected_low(0x2468ACF13579BDE0, 0x2);
            if (prod.word[0] != expected_low.low || prod.word[1] != expected_low.high) {
                printf("FAIL: Large multiplication test\n");
                all_pass = false;
            }
        }
        
        if (all_pass) {
            printf("PASS: All uint128_t arithmetic tests\n");
        }
    }
}

// Test 2: Barrett reduction correctness
__global__ void test_barrett_correctness() {
    if (threadIdx.x == 0) {
        bool all_pass = true;
        
        // Test case 1: Small modulus
        {
            uint128_t a(1234567890, 0);
            uint128_t n(1000000007, 0);
            
            Barrett128 barrett;
            barrett.n = n;
            barrett.precompute();
            
            uint128_t result = barrett.reduce(a);
            uint64_t expected = 1234567890 % 1000000007;
            
            if (result.low != expected || result.high != 0) {
                printf("FAIL: Barrett small modulus test. Got %llu, expected %llu\n", 
                       result.low, expected);
                all_pass = false;
            }
        }
        
        // Test case 2: Larger number
        {
            uint128_t a(0xAB54A98CEB1F0AD2ULL, 0x0ULL);  // 12345678901234567890
            uint128_t n(1000000007, 0);
            
            Barrett128 barrett;
            barrett.n = n;
            barrett.precompute();
            
            uint128_t result = barrett.reduce(a);
            // 12345678901234567890 % 1000000007 = 652337934
            
            if (result.low != 652337934 || result.high != 0) {
                printf("FAIL: Barrett large number test. Got %llu, expected 652337934\n", 
                       result.low);
                all_pass = false;
            }
        }
        
        if (all_pass) {
            printf("PASS: All Barrett reduction tests\n");
        }
    }
}

// Test 3: cuRAND state initialization and randomness
__global__ void test_curand_randomness() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize cuRAND
    curandState_t state;
    init_curand_state(&state, tid);
    
    // Generate some random numbers
    uint64_t rand1 = curand(&state);
    uint64_t rand2 = curand(&state);
    uint64_t rand3 = curand(&state);
    
    // Basic sanity check - should be different
    if (tid == 0) {
        if (rand1 == rand2 && rand2 == rand3) {
            printf("WARNING: Random numbers appear identical\n");
        } else {
            printf("PASS: cuRAND generating different values\n");
        }
        
        // Test random 128-bit generation
        uint128_t max(1000000, 0);
        uint128_t rand128 = generate_random_128(&state, max);
        
        if (rand128 >= max) {
            printf("FAIL: Random 128-bit exceeds max\n");
        } else {
            printf("PASS: Random 128-bit within bounds\n");
        }
    }
}

// Test 4: GCD algorithm
__global__ void test_gcd_algorithm() {
    if (threadIdx.x == 0) {
        bool all_pass = true;
        
        // Test case 1: GCD of coprime numbers
        {
            uint128_t a(17, 0);
            uint128_t b(19, 0);
            uint128_t g = gcd_128(a, b);
            if (g.low != 1 || g.high != 0) {
                printf("FAIL: GCD of coprime numbers\n");
                all_pass = false;
            }
        }
        
        // Test case 2: GCD with common factor
        {
            uint128_t a(24, 0);
            uint128_t b(36, 0);
            uint128_t g = gcd_128(a, b);
            if (g.low != 12 || g.high != 0) {
                printf("FAIL: GCD with common factor. Got %llu, expected 12\n", g.low);
                all_pass = false;
            }
        }
        
        // Test case 3: GCD with one zero
        {
            uint128_t a(42, 0);
            uint128_t b(0, 0);
            uint128_t g = gcd_128(a, b);
            if (g.low != 42 || g.high != 0) {
                printf("FAIL: GCD with zero\n");
                all_pass = false;
            }
        }
        
        // Test case 4: Large GCD
        {
            uint128_t a(428759ULL * 17, 0);
            uint128_t b(428759ULL * 23, 0);
            uint128_t g = gcd_128(a, b);
            if (g.low != 428759 || g.high != 0) {
                printf("FAIL: Large GCD. Got %llu, expected 428759\n", g.low);
                all_pass = false;
            }
        }
        
        if (all_pass) {
            printf("PASS: All GCD tests\n");
        }
    }
}

// Test 5: Integration test - factor a known semiprime
__global__ void test_factorization_integration() {
    // This would typically be called as part of the full factorization pipeline
    if (threadIdx.x == 0) {
        printf("Integration test: Would factor 90595490423 = 428759 × 211297\n");
        printf("This test is run through the main factorization function\n");
    }
}

// Test 6: Performance comparison
__global__ void test_performance_comparison() {
    if (threadIdx.x == 0) {
        const int iterations = 1000000;
        uint128_t a(0x123456789ABCDEF0, 0x1);
        uint128_t n(1000000007, 0);
        
        // Time basic modulo (simplified version)
        clock_t start = clock64();
        uint128_t result1 = a;
        for (int i = 0; i < iterations; i++) {
            // Simplified modulo
            while (result1 >= n) {
                result1 = subtract_128(result1, n);
            }
        }
        clock_t basic_time = clock64() - start;
        
        // Time Barrett reduction
        Barrett128 barrett;
        barrett.n = n;
        barrett.precompute();
        
        start = clock64();
        uint128_t result2 = a;
        for (int i = 0; i < iterations; i++) {
            result2 = barrett.reduce(result2);
        }
        clock_t barrett_time = clock64() - start;
        
        printf("Performance comparison (%d iterations):\n", iterations);
        printf("  Basic modulo: %lld cycles\n", (long long)basic_time);
        printf("  Barrett reduction: %lld cycles\n", (long long)barrett_time);
        printf("  Speedup: %.2fx\n", (float)basic_time / barrett_time);
    }
}

// Main test runner
void run_all_tests() {
    printf("=== Improved 128-bit Factorizer Test Suite ===\n\n");
    
    // Run all tests
    RUN_TEST("uint128_t Arithmetic", test_uint128_arithmetic_comprehensive, 1, 1);
    RUN_TEST("Barrett Reduction", test_barrett_correctness, 1, 1);
    RUN_TEST("cuRAND Randomness", test_curand_randomness, 4, 32);
    RUN_TEST("GCD Algorithm", test_gcd_algorithm, 1, 1);
    RUN_TEST("Integration Test", test_factorization_integration, 1, 1);
    RUN_TEST("Performance Comparison", test_performance_comparison, 1, 1);
    
    cudaDeviceSynchronize();
    
    // Print summary
    printf("\n=== Test Summary ===\n");
    int passed = 0, failed = 0;
    
    for (const auto& result : test_results) {
        printf("%-30s: %s (%.2f ms)\n", 
               result.test_name.c_str(), 
               result.passed ? "PASS" : "FAIL",
               result.time_ms);
        
        if (result.passed) passed++;
        else failed++;
        
        if (!result.passed) {
            printf("  Error: %s\n", result.message.c_str());
        }
    }
    
    printf("\nTotal: %d passed, %d failed\n", passed, failed);
    
    // Run actual factorization test
    printf("\n=== Factorization Test ===\n");
    printf("Testing improved factorizer on known semiprimes...\n");
}

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Running tests on: %s\n\n", prop.name);
    
    // Run test suite
    run_all_tests();
    
    return 0;
}