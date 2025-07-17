/**
 * Comprehensive Test Suite for Version 2.1.0 Features
 * Tests Barrett v2, Montgomery, cuRAND v2, and Progress Monitoring
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>

#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include "montgomery_reduction.cuh"
#include "curand_pollards_rho_v2.cuh"
#include "progress_monitor_fixed.cuh"

// Test result tracking
struct TestResult {
    std::string test_name;
    bool passed;
    double time_ms;
    std::string error_message;
};

std::vector<TestResult> test_results;

// Helper to run and track tests
#define RUN_TEST(test_func, name) do { \
    printf("\n=== Running: %s ===\n", name); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    bool passed = test_func(); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float ms = 0; \
    cudaEventElapsedTime(&ms, start, stop); \
    TestResult result = {name, passed, ms, passed ? "" : "Test failed"}; \
    test_results.push_back(result); \
    printf("Result: %s (%.2f ms)\n", passed ? "PASSED" : "FAILED", ms); \
} while(0)

// Test 1: Barrett Reduction v2
bool test_barrett_v2() {
    // Test cases with known results
    struct TestCase {
        uint128_t a;
        uint128_t n;
        uint128_t expected;
    };
    
    TestCase cases[] = {
        // a, modulus, expected result
        {uint128_t(12345678901234567890ULL, 0), uint128_t(1000000007, 0), uint128_t(652337934, 0)},
        {uint128_t(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL), uint128_t(0x1234567890ABCDEFULL, 0), uint128_t(0x123456789098B4EULL, 0)},
        {uint128_t(999999999999999999ULL, 0), uint128_t(17, 0), uint128_t(4, 0)}
    };
    
    for (int i = 0; i < 3; i++) {
        Barrett128_v2 barrett;
        barrett.n = cases[i].n;
        barrett.precompute();
        
        uint128_t result = barrett.reduce_128(cases[i].a);
        
        if (result != cases[i].expected) {
            printf("Barrett test case %d failed: got %llx:%llx, expected %llx:%llx\n",
                   i, result.high, result.low, cases[i].expected.high, cases[i].expected.low);
            return false;
        }
    }
    
    // Test modular multiplication
    uint128_t a(12345, 0);
    uint128_t b(67890, 0);
    uint128_t n(1000000007, 0);
    
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    uint128_t result = modmul_barrett_v2(a, b, barrett);
    uint128_t expected(838102050, 0);  // (12345 * 67890) % 1000000007
    
    if (result != expected) {
        printf("Barrett modmul failed: got %llu, expected %llu\n", result.low, expected.low);
        return false;
    }
    
    printf("All Barrett v2 tests passed!\n");
    return true;
}

// Test 2: Montgomery Reduction
bool test_montgomery() {
    // Test with odd modulus
    uint128_t n(1000000007, 0);  // Prime
    
    Montgomery128 mont;
    mont.n = n;
    mont.precompute();
    
    // Test 1: Conversion to/from Montgomery form
    uint128_t a(12345, 0);
    uint128_t a_mont = to_montgomery(a, mont);
    uint128_t a_back = from_montgomery(a_mont, mont);
    
    if (a != a_back) {
        printf("Montgomery conversion failed: %llu -> %llu\n", a.low, a_back.low);
        return false;
    }
    
    // Test 2: Multiplication
    uint128_t b(67890, 0);
    uint128_t expected(838102050, 0);  // (12345 * 67890) % 1000000007
    
    uint128_t a_mont2 = to_montgomery(a, mont);
    uint128_t b_mont = to_montgomery(b, mont);
    uint128_t result_mont = montgomery_multiply(a_mont2, b_mont, mont);
    uint128_t result = from_montgomery(result_mont, mont);
    
    if (result != expected) {
        printf("Montgomery multiply failed: got %llu, expected %llu\n", result.low, expected.low);
        return false;
    }
    
    // Test 3: Exponentiation
    uint128_t base(2, 0);
    uint128_t exp(10, 0);
    uint128_t exp_result = montgomery_exponent(base, exp, mont);
    uint128_t exp_expected(1024, 0);  // 2^10 = 1024
    
    if (exp_result != exp_expected) {
        printf("Montgomery exponent failed: got %llu, expected %llu\n", exp_result.low, exp_expected.low);
        return false;
    }
    
    printf("All Montgomery tests passed!\n");
    return true;
}

// Test 3: cuRAND v2 Integration
__global__ void test_curand_kernel(bool* success) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize cuRAND
    curandState_t state;
    int error_code;
    
    if (!init_curand_state_safe(&state, tid, &error_code)) {
        *success = false;
        return;
    }
    
    // Test random generation
    uint128_t min(100, 0);
    uint128_t max(1000, 0);
    
    bool local_success = true;
    for (int i = 0; i < 100; i++) {
        uint128_t rand = generate_random_128_safe(&state, min, max, &error_code);
        if (error_code != CURAND_SUCCESS || rand < min || rand >= max) {
            local_success = false;
            break;
        }
    }
    
    if (!local_success) {
        *success = false;
    }
}

bool test_curand_v2() {
    bool* d_success;
    cudaMalloc(&d_success, sizeof(bool));
    
    bool h_success = true;
    cudaMemcpy(d_success, &h_success, sizeof(bool), cudaMemcpyHostToDevice);
    
    // Launch test kernel
    test_curand_kernel<<<32, 256>>>(d_success);
    
    cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_success);
    
    if (!h_success) {
        printf("cuRAND v2 test failed\n");
        return false;
    }
    
    // Test factorization with cuRAND
    uint128_t n(90595490423ULL, 0);  // 428759 × 211297
    FactorizationResult result;
    
    launch_pollards_rho_v2(n, &result, 16, 256, 100000, false, false);
    
    if (result.factor_count < 2) {
        printf("cuRAND factorization failed: found %d factors\n", result.factor_count);
        return false;
    }
    
    printf("cuRAND v2 factorization succeeded: found %d factors in %.2f ms\n",
           result.factor_count, result.time_ms);
    
    for (int i = 0; i < result.factor_count; i++) {
        printf("  Factor %d: %llu\n", i+1, result.factors[i].low);
    }
    
    return true;
}

// Test 4: Progress Monitoring
bool test_progress_monitor() {
    // Create a simple test kernel that updates progress
    auto test_kernel = []__global__(ProgressState* progress) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        for (int i = 0; i < 1000; i++) {
            update_progress_device(progress, 1, 1, 1);
            
            // Simulate some work
            uint128_t x(tid + i, tid);
            uint128_t n(1000000007, 0);
            uint128_t g = gcd_128(x, n);
        }
    };
    
    // Test progress tracking
    uint128_t target(0x123456789ABCDEFULL, 0x1ULL);
    ProgressReporter reporter(target, 256, false);  // Disable verbose output for test
    
    // Launch kernel
    test_kernel<<<1, 256>>>(reporter.get_device_pointer());
    cudaDeviceSynchronize();
    
    // Update and check metrics
    reporter.update_and_report();
    
    // Verify progress was tracked
    // (We can't easily verify exact values due to timing, but we can check non-zero)
    ProgressState h_progress;
    cudaMemcpy(&h_progress, reporter.get_device_pointer(), sizeof(ProgressState), cudaMemcpyDeviceToHost);
    
    if (h_progress.total_iterations == 0) {
        printf("Progress monitoring failed: no iterations recorded\n");
        return false;
    }
    
    printf("Progress monitoring successful: %llu iterations tracked\n",
           (unsigned long long)h_progress.total_iterations);
    
    return true;
}

// Test 5: Performance Comparison
bool test_performance_comparison() {
    printf("\nPerformance Comparison (1000 modular multiplications):\n");
    
    uint128_t n(0xFFFFFFFFFFFFFFC5ULL, 0xFFFFFFFFFFFFFFFFULL);  // Large prime
    uint128_t a(0x123456789ABCDEFULL, 0x1ULL);
    uint128_t b(0xFEDCBA9876543210ULL, 0x2ULL);
    
    // Test Barrett v2
    {
        Barrett128_v2 barrett;
        barrett.n = n;
        barrett.precompute();
        
        clock_t start = clock();
        uint128_t result = a;
        for (int i = 0; i < 1000; i++) {
            result = modmul_barrett_v2(result, b, barrett);
        }
        clock_t end = clock();
        
        double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        printf("  Barrett v2: %.3f ms\n", time_ms);
    }
    
    // Test Montgomery (only for odd modulus)
    if (n.low & 1) {
        Montgomery128 mont;
        mont.n = n;
        mont.precompute();
        
        uint128_t a_mont = to_montgomery(a, mont);
        uint128_t b_mont = to_montgomery(b, mont);
        
        clock_t start = clock();
        uint128_t result = a_mont;
        for (int i = 0; i < 1000; i++) {
            result = montgomery_multiply(result, b_mont, mont);
        }
        result = from_montgomery(result, mont);
        clock_t end = clock();
        
        double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        printf("  Montgomery: %.3f ms\n", time_ms);
    }
    
    return true;
}

// Test 6: Integration Test - Complete Factorization
bool test_complete_factorization() {
    printf("\nTesting complete factorization with all v2.1.0 features:\n");
    
    // Test number: 30-digit semiprime
    uint128_t n(1234567890123456789ULL, 67890123ULL);
    
    // Create progress reporter
    int total_threads = 64 * 256;
    ProgressReporter reporter(n, total_threads, true);
    
    // Allocate memory for results
    FactorizationResult* d_result;
    cudaMalloc(&d_result, sizeof(FactorizationResult));
    cudaMemset(d_result, 0, sizeof(FactorizationResult));
    
    printf("Starting factorization of %llx:%llx\n", n.high, n.low);
    
    // Launch with all optimizations enabled
    pollards_rho_curand_v2<<<64, 256>>>(
        n, d_result, 1000000,
        true,  // Use Montgomery
        true   // Use Brent's variant
    );
    
    // Monitor progress for up to 30 seconds
    clock_t start_time = clock();
    while ((clock() - start_time) / CLOCKS_PER_SEC < 30) {
        reporter.update_and_report();
        
        if (cudaStreamQuery(0) == cudaSuccess) {
            break;
        }
        
        usleep(500000);  // 500ms
    }
    
    // Get results
    FactorizationResult h_result;
    cudaMemcpy(&h_result, d_result, sizeof(FactorizationResult), cudaMemcpyDeviceToHost);
    
    printf("\nFactorization complete:\n");
    printf("  Factors found: %d\n", h_result.factor_count);
    printf("  Total iterations: %d\n", h_result.total_iterations);
    printf("  Successful threads: %d\n", h_result.successful_threads);
    printf("  Time: %.2f ms\n", h_result.time_ms);
    
    cudaFree(d_result);
    
    return h_result.factor_count > 0;
}

// Main test runner
int main(int argc, char** argv) {
    printf("=================================\n");
    printf("Version 2.1.0 Feature Test Suite\n");
    printf("=================================\n");
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Run all tests
    RUN_TEST(test_barrett_v2, "Barrett Reduction v2");
    RUN_TEST(test_montgomery, "Montgomery Reduction");
    RUN_TEST(test_curand_v2, "cuRAND v2 Integration");
    RUN_TEST(test_progress_monitor, "Progress Monitoring");
    RUN_TEST(test_performance_comparison, "Performance Comparison");
    RUN_TEST(test_complete_factorization, "Complete Factorization");
    
    // Summary
    printf("\n=================================\n");
    printf("Test Summary:\n");
    printf("=================================\n");
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& result : test_results) {
        printf("%-30s: %s (%.2f ms)\n", 
               result.test_name.c_str(),
               result.passed ? "PASSED" : "FAILED",
               result.time_ms);
        
        if (result.passed) passed++;
        else failed++;
    }
    
    printf("\nTotal: %d passed, %d failed\n", passed, failed);
    
    return (failed == 0) ? 0 : 1;
}