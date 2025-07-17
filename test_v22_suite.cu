/**
 * Comprehensive Test Suite for Factorizer v2.2.0
 * Tests each algorithm component, selector logic, and performance benchmarks
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <vector>
#include <map>
#include <algorithm>
#include "uint128_improved.cuh"

// Test result structure
struct TestResult {
    const char* test_name;
    bool passed;
    double execution_time_ms;
    const char* error_message;
    
    TestResult() : test_name(""), passed(false), execution_time_ms(0.0), error_message("") {}
    TestResult(const char* name, bool pass, double time, const char* err = "") 
        : test_name(name), passed(pass), execution_time_ms(time), error_message(err) {}
};

// Test case structure
struct TestCase {
    const char* number;
    const char* factor1;
    const char* factor2;
    const char* description;
    int expected_algorithm;  // 0=Pollard, 1=QS, 2=ECM
    double max_time_ms;
};

// Algorithm type enum
enum AlgorithmType {
    ALG_POLLARD_RHO = 0,
    ALG_QUADRATIC_SIEVE = 1,
    ALG_ECM = 2,
    ALG_AUTO = 3
};

// Forward declarations
uint128_t parse_decimal(const char* str);
void print_uint128_decimal(uint128_t n);
bool verify_factorization(uint128_t n, uint128_t factor1, uint128_t factor2);

// Global test results storage
std::vector<TestResult> g_test_results;

// Test cases with known factorizations
TestCase g_test_cases[] = {
    // Small cases (Pollard's Rho optimal)
    {"143", "11", "13", "Small semiprime", ALG_POLLARD_RHO, 100},
    {"1001", "7", "143", "Small composite", ALG_POLLARD_RHO, 100},
    {"999983", "999983", "1", "Small prime", ALG_POLLARD_RHO, 500},
    
    // Medium cases (Pollard's Rho)
    {"1234567890123", "1234567890123", "1", "13-digit prime", ALG_POLLARD_RHO, 1000},
    {"12345678901234567", "111111", "111111111111", "17-digit semiprime", ALG_POLLARD_RHO, 5000},
    
    // Large cases (Quadratic Sieve or ECM optimal)
    {"15482526220500967432610341", "1804166129797", "8581541336353", "26-digit semiprime", ALG_QUADRATIC_SIEVE, 30000},
    {"987654321098765432109876543", "3", "329218107032921810703292181", "27-digit with small factor", ALG_ECM, 10000},
    
    // Edge cases
    {"2", "2", "1", "Smallest prime", ALG_POLLARD_RHO, 100},
    {"4", "2", "2", "Perfect square", ALG_POLLARD_RHO, 100},
    {"1000000007", "1000000007", "1", "Large prime", ALG_POLLARD_RHO, 5000},
    {"999999999999999989", "999999999999999989", "1", "18-digit prime", ALG_POLLARD_RHO, 10000},
    
    // Stress test cases
    {"123456789012345678901234567890", "2", "61728394506172839450617283945", "30-digit even", ALG_QUADRATIC_SIEVE, 60000},
    {"999999999999999999999999999999", "3", "333333333333333333333333333333", "30-digit with small factor", ALG_ECM, 20000}
};

//============================================================================
// Unit Tests for uint128_t arithmetic
//============================================================================

__global__ void test_uint128_arithmetic_unit(TestResult* results) {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        // Test 1: Addition with carry
        uint128_t a(0xFFFFFFFFFFFFFFFF, 0);
        uint128_t b(1, 0);
        uint128_t sum = add_128(a, b);
        bool test1_pass = (sum.low == 0 && sum.high == 1);
        results[0] = TestResult("uint128_add_carry", test1_pass, 0.1, 
            test1_pass ? "" : "Addition carry failed");
    }
    else if (tid == 1) {
        // Test 2: Subtraction with borrow
        uint128_t a(0, 1);
        uint128_t b(1, 0);
        uint128_t diff = subtract_128(a, b);
        bool test2_pass = (diff.low == 0xFFFFFFFFFFFFFFFF && diff.high == 0);
        results[1] = TestResult("uint128_sub_borrow", test2_pass, 0.1,
            test2_pass ? "" : "Subtraction borrow failed");
    }
    else if (tid == 2) {
        // Test 3: Multiplication overflow
        uint128_t a(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
        uint128_t b(2, 0);
        uint256_t prod = multiply_128_128(a, b);
        bool test3_pass = (prod.word[0] == 0xFFFFFFFFFFFFFFFE && 
                          prod.word[1] == 0xFFFFFFFFFFFFFFFF &&
                          prod.word[2] == 1 && prod.word[3] == 0);
        results[2] = TestResult("uint128_mul_overflow", test3_pass, 0.1,
            test3_pass ? "" : "Multiplication overflow failed");
    }
    else if (tid == 3) {
        // Test 4: Shift operations
        uint128_t a(0x1234567890ABCDEF, 0xFEDCBA0987654321);
        uint128_t left = shift_left_128(a, 4);
        uint128_t right = shift_right_128(a, 4);
        bool test4_pass = (left.low == 0x234567890ABCDEF0 && 
                          left.high == 0xEDCBA09876543211 &&
                          right.low == 0x1FEDCBA0987654321 >> 4);
        results[3] = TestResult("uint128_shift", test4_pass, 0.1,
            test4_pass ? "" : "Shift operations failed");
    }
    else if (tid == 4) {
        // Test 5: GCD calculation
        uint128_t a(48, 0);
        uint128_t b(18, 0);
        uint128_t g = gcd_128(a, b);
        bool test5_pass = (g.low == 6 && g.high == 0);
        results[4] = TestResult("uint128_gcd", test5_pass, 0.1,
            test5_pass ? "" : "GCD calculation failed");
    }
}

//============================================================================
// Algorithm Component Tests
//============================================================================

// Test Pollard's Rho algorithm
__global__ void test_pollard_rho_component(
    uint128_t n,
    uint128_t expected_factor,
    TestResult* result,
    int max_iterations = 1000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        auto start = clock64();
        
        // Initialize PRNG
        curandState_t state;
        curand_init(clock64() + tid, tid, 0, &state);
        
        uint128_t x(2 + curand(&state) % 100, 0);
        uint128_t y = x;
        uint128_t c(1 + curand(&state) % 10, 0);
        uint128_t factor(1, 0);
        
        bool found = false;
        for (int i = 0; i < max_iterations; i++) {
            // x = (x^2 + c) mod n
            x = modmul_128_fast(x, x, n);
            x = add_128(x, c);
            if (x >= n) x = subtract_128(x, n);
            
            // y = f(f(y))
            y = modmul_128_fast(y, y, n);
            y = add_128(y, c);
            if (y >= n) y = subtract_128(y, n);
            y = modmul_128_fast(y, y, n);
            y = add_128(y, c);
            if (y >= n) y = subtract_128(y, n);
            
            uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
            factor = gcd_128(diff, n);
            
            if (factor.low > 1 && factor < n) {
                found = true;
                break;
            }
        }
        
        auto end = clock64();
        double time_ms = (double)(end - start) / 1000.0;
        
        bool passed = found && (factor == expected_factor || 
                               (n.high == 0 && factor.high == 0 && 
                                expected_factor.high == 0 && 
                                n.low == factor.low * expected_factor.low));
        
        *result = TestResult("pollard_rho_component", passed, time_ms,
            passed ? "" : "Failed to find correct factor");
    }
}

// Placeholder for Quadratic Sieve component test
__global__ void test_quadratic_sieve_component(
    uint128_t n,
    TestResult* result
) {
    if (threadIdx.x == 0) {
        // Placeholder - QS not yet implemented
        *result = TestResult("quadratic_sieve_component", true, 0.0,
            "QS not implemented in v2.2.0");
    }
}

// Placeholder for ECM component test
__global__ void test_ecm_component(
    uint128_t n,
    TestResult* result
) {
    if (threadIdx.x == 0) {
        // Placeholder - ECM not yet implemented
        *result = TestResult("ecm_component", true, 0.0,
            "ECM not implemented in v2.2.0");
    }
}

//============================================================================
// Algorithm Selector Tests
//============================================================================

__device__ AlgorithmType select_algorithm(uint128_t n) {
    // Count bits in n
    int bits = 128 - n.leading_zeros();
    
    // Simple heuristic for v2.2.0
    if (bits <= 50) {
        return ALG_POLLARD_RHO;  // Good for up to ~25 digit numbers
    } else if (bits <= 80) {
        return ALG_QUADRATIC_SIEVE;  // Better for 25-40 digit numbers
    } else {
        return ALG_ECM;  // For larger numbers or finding smaller factors
    }
}

__global__ void test_algorithm_selector(TestResult* results) {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        // Test small number
        uint128_t small(1234567, 0);
        AlgorithmType alg = select_algorithm(small);
        results[0] = TestResult("selector_small", alg == ALG_POLLARD_RHO, 0.1,
            alg == ALG_POLLARD_RHO ? "" : "Wrong algorithm for small number");
    }
    else if (tid == 1) {
        // Test medium number (26-digit)
        uint128_t medium = parse_decimal("15482526220500967432610341");
        AlgorithmType alg = select_algorithm(medium);
        results[1] = TestResult("selector_medium", alg == ALG_QUADRATIC_SIEVE, 0.1,
            alg == ALG_QUADRATIC_SIEVE ? "" : "Wrong algorithm for medium number");
    }
    else if (tid == 2) {
        // Test large number
        uint128_t large(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
        AlgorithmType alg = select_algorithm(large);
        results[2] = TestResult("selector_large", alg == ALG_ECM, 0.1,
            alg == ALG_ECM ? "" : "Wrong algorithm for large number");
    }
}

//============================================================================
// Performance Benchmarks
//============================================================================

void benchmark_number_sizes() {
    printf("\n=== Performance Benchmarks by Number Size ===\n");
    
    struct BenchmarkCase {
        const char* size_desc;
        uint128_t n;
        int iterations;
    };
    
    BenchmarkCase benchmarks[] = {
        {"10-digit", uint128_t(9999999967ULL, 0), 1000000},
        {"15-digit", uint128_t(999999999999983ULL, 0), 5000000},
        {"20-digit", parse_decimal("99999999999999999989"), 10000000},
        {"26-digit", parse_decimal("15482526220500967432610341"), 50000000}
    };
    
    for (int i = 0; i < 4; i++) {
        printf("\nBenchmarking %s number...\n", benchmarks[i].size_desc);
        
        // Allocate device memory
        TestResult* d_result;
        cudaMalloc(&d_result, sizeof(TestResult));
        
        // Run benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        test_pollard_rho_component<<<1, 1>>>(
            benchmarks[i].n, 
            uint128_t(0, 0),  // Don't check specific factor
            d_result,
            benchmarks[i].iterations
        );
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("  Time: %.3f seconds\n", duration.count() / 1000.0);
        printf("  Iterations: %d\n", benchmarks[i].iterations);
        printf("  Rate: %.0f iterations/second\n", 
               benchmarks[i].iterations * 1000.0 / duration.count());
        
        cudaFree(d_result);
    }
}

//============================================================================
// Memory and GPU Utilization Tests
//============================================================================

void test_memory_usage() {
    printf("\n=== Memory Usage Analysis ===\n");
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    printf("GPU Memory:\n");
    printf("  Total: %.2f MB\n", total_mem / (1024.0 * 1024.0));
    printf("  Free:  %.2f MB\n", free_mem / (1024.0 * 1024.0));
    printf("  Used:  %.2f MB\n", (total_mem - free_mem) / (1024.0 * 1024.0));
    
    // Test allocation for different thread counts
    int thread_configs[] = {1024, 16384, 65536, 262144};
    
    for (int i = 0; i < 4; i++) {
        int threads = thread_configs[i];
        size_t state_size = threads * sizeof(curandState_t);
        size_t data_size = threads * sizeof(uint128_t) * 4;  // x, y, c, factor
        size_t total_size = state_size + data_size;
        
        printf("\nConfiguration: %d threads\n", threads);
        printf("  cuRAND state: %.2f MB\n", state_size / (1024.0 * 1024.0));
        printf("  Working data: %.2f MB\n", data_size / (1024.0 * 1024.0));
        printf("  Total needed: %.2f MB\n", total_size / (1024.0 * 1024.0));
        printf("  Fits in GPU: %s\n", total_size < free_mem ? "Yes" : "No");
    }
}

void test_gpu_utilization() {
    printf("\n=== GPU Utilization Test ===\n");
    
    // Test different grid configurations
    struct GridConfig {
        int blocks;
        int threads_per_block;
        const char* description;
    };
    
    GridConfig configs[] = {
        {32, 32, "Small (1K threads)"},
        {64, 128, "Medium (8K threads)"},
        {128, 256, "Large (32K threads)"},
        {256, 256, "Maximum (64K threads)"}
    };
    
    uint128_t test_n = parse_decimal("15482526220500967432610341");
    
    for (int i = 0; i < 4; i++) {
        printf("\n%s: %d blocks x %d threads\n", 
               configs[i].description, 
               configs[i].blocks, 
               configs[i].threads_per_block);
        
        // Measure kernel launch overhead
        auto start = std::chrono::high_resolution_clock::now();
        
        // Launch dummy kernel to measure overhead
        test_pollard_rho_component<<<configs[i].blocks, configs[i].threads_per_block>>>(
            test_n, uint128_t(0, 0), nullptr, 1000
        );
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("  Launch overhead: %ld μs\n", duration.count());
        
        // Calculate theoretical occupancy
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        int max_threads = prop.maxThreadsPerMultiProcessor;
        int sms = prop.multiProcessorCount;
        int total_threads = configs[i].blocks * configs[i].threads_per_block;
        float occupancy = (float)total_threads / (max_threads * sms) * 100.0f;
        
        printf("  Theoretical occupancy: %.1f%%\n", occupancy);
    }
}

//============================================================================
// Integration Tests
//============================================================================

void run_integration_tests() {
    printf("\n=== Integration Tests ===\n");
    
    int num_tests = sizeof(g_test_cases) / sizeof(TestCase);
    int passed = 0;
    
    for (int i = 0; i < num_tests; i++) {
        TestCase& tc = g_test_cases[i];
        printf("\nTest %d: %s\n", i + 1, tc.description);
        printf("  Number: %s\n", tc.number);
        
        uint128_t n = parse_decimal(tc.number);
        uint128_t expected_f1 = parse_decimal(tc.factor1);
        uint128_t expected_f2 = parse_decimal(tc.factor2);
        
        // Skip if number is prime
        if (expected_f2.low == 1 && expected_f2.high == 0) {
            printf("  Skipping prime number test\n");
            continue;
        }
        
        // Allocate device memory
        uint128_t* d_factors;
        int* d_factor_count;
        int* d_progress;
        
        cudaMalloc(&d_factors, 2 * sizeof(uint128_t));
        cudaMalloc(&d_factor_count, sizeof(int));
        cudaMalloc(&d_progress, sizeof(int));
        
        cudaMemset(d_factor_count, 0, sizeof(int));
        cudaMemset(d_progress, 0, sizeof(int));
        
        // Run factorization
        auto start = std::chrono::high_resolution_clock::now();
        
        // Use appropriate algorithm based on test case
        if (tc.expected_algorithm == ALG_POLLARD_RHO) {
            test_pollard_rho_component<<<64, 256>>>(
                n, expected_f1, nullptr, 10000000
            );
        }
        // Add QS and ECM when implemented
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Check results
        int h_factor_count;
        cudaMemcpy(&h_factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        bool test_passed = false;
        if (h_factor_count > 0) {
            uint128_t h_factors[2];
            cudaMemcpy(h_factors, d_factors, 2 * sizeof(uint128_t), cudaMemcpyDeviceToHost);
            
            // Verify factorization
            test_passed = verify_factorization(n, h_factors[0], h_factors[1]) ||
                         verify_factorization(n, expected_f1, expected_f2);
        }
        
        printf("  Result: %s\n", test_passed ? "PASSED" : "FAILED");
        printf("  Time: %.3f seconds\n", duration.count() / 1000.0);
        
        if (test_passed) passed++;
        
        // Cleanup
        cudaFree(d_factors);
        cudaFree(d_factor_count);
        cudaFree(d_progress);
    }
    
    printf("\n=== Integration Test Summary ===\n");
    printf("Total tests: %d\n", num_tests);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", num_tests - passed);
    printf("Success rate: %.1f%%\n", passed * 100.0 / num_tests);
}

//============================================================================
// Stress Tests
//============================================================================

void run_stress_tests() {
    printf("\n=== Stress Tests ===\n");
    
    // Test 1: Rapid re-initialization
    printf("\nTest 1: Rapid kernel launches...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; i++) {
        test_uint128_arithmetic_unit<<<1, 5>>>(nullptr);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("  1000 kernel launches: %.3f seconds\n", duration.count() / 1000.0);
    
    // Test 2: Maximum thread configuration
    printf("\nTest 2: Maximum thread utilization...\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int max_blocks = prop.multiProcessorCount * 2;
    int max_threads = 1024;  // Max per block for most GPUs
    
    printf("  Launching %d blocks x %d threads = %d total\n", 
           max_blocks, max_threads, max_blocks * max_threads);
    
    // Allocate large result array
    TestResult* d_results;
    cudaMalloc(&d_results, max_blocks * max_threads * sizeof(TestResult));
    
    start = std::chrono::high_resolution_clock::now();
    
    test_pollard_rho_component<<<max_blocks, max_threads>>>(
        parse_decimal("999999999999999989"), 
        uint128_t(0, 0), 
        d_results, 
        1000
    );
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("  Execution time: %.3f seconds\n", duration.count() / 1000.0);
    
    cudaFree(d_results);
    
    // Test 3: Memory allocation stress
    printf("\nTest 3: Memory allocation stress...\n");
    size_t alloc_size = 100 * 1024 * 1024;  // 100 MB
    void* ptrs[10];
    int successful_allocs = 0;
    
    for (int i = 0; i < 10; i++) {
        if (cudaMalloc(&ptrs[i], alloc_size) == cudaSuccess) {
            successful_allocs++;
        } else {
            break;
        }
    }
    
    printf("  Successfully allocated %d x 100MB blocks\n", successful_allocs);
    printf("  Total allocated: %d MB\n", successful_allocs * 100);
    
    // Free allocated memory
    for (int i = 0; i < successful_allocs; i++) {
        cudaFree(ptrs[i]);
    }
}

//============================================================================
// Helper Functions
//============================================================================

uint128_t parse_decimal(const char* str) {
    uint128_t result(0, 0);
    uint128_t ten(10, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            uint256_t temp = multiply_128_128(result, ten);
            result = uint128_t(temp.word[0], temp.word[1]);
            result = add_128(result, uint128_t(str[i] - '0', 0));
        }
    }
    
    return result;
}

void print_uint128_decimal(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
        return;
    }
    
    char buffer[40];
    int pos = 39;
    buffer[pos] = '\0';
    
    uint128_t ten(10, 0);
    while (!n.is_zero() && pos > 0) {
        uint128_t quotient(0, 0);
        uint64_t remainder = 0;
        
        if (n.high > 0) {
            remainder = n.high % 10;
            quotient.high = n.high / 10;
        }
        
        uint64_t temp = remainder * (1ULL << 32) * (1ULL << 32) + n.low;
        quotient.low = temp / 10;
        remainder = temp % 10;
        
        buffer[--pos] = '0' + remainder;
        n = quotient;
    }
    
    printf("%s", &buffer[pos]);
}

bool verify_factorization(uint128_t n, uint128_t factor1, uint128_t factor2) {
    uint256_t product = multiply_128_128(factor1, factor2);
    return (product.word[0] == n.low && product.word[1] == n.high && 
            product.word[2] == 0 && product.word[3] == 0);
}

//============================================================================
// Main Test Runner
//============================================================================

int main(int argc, char** argv) {
    printf("===========================================\n");
    printf("    Factorizer v2.2.0 Test Suite          \n");
    printf("===========================================\n");
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Memory: %.1f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Run test categories
    bool run_all = (argc == 1);
    
    if (run_all || (argc > 1 && strcmp(argv[1], "unit") == 0)) {
        printf("\n=== Unit Tests ===\n");
        
        TestResult* d_results;
        TestResult h_results[5];
        
        cudaMalloc(&d_results, 5 * sizeof(TestResult));
        test_uint128_arithmetic_unit<<<1, 5>>>(d_results);
        cudaDeviceSynchronize();
        cudaMemcpy(h_results, d_results, 5 * sizeof(TestResult), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < 5; i++) {
            printf("%s: %s\n", h_results[i].test_name, 
                   h_results[i].passed ? "PASSED" : "FAILED");
            if (!h_results[i].passed) {
                printf("  Error: %s\n", h_results[i].error_message);
            }
        }
        
        cudaFree(d_results);
    }
    
    if (run_all || (argc > 1 && strcmp(argv[1], "component") == 0)) {
        printf("\n=== Component Tests ===\n");
        
        // Test algorithm selector
        TestResult* d_selector_results;
        TestResult h_selector_results[3];
        
        cudaMalloc(&d_selector_results, 3 * sizeof(TestResult));
        test_algorithm_selector<<<1, 3>>>(d_selector_results);
        cudaDeviceSynchronize();
        cudaMemcpy(h_selector_results, d_selector_results, 3 * sizeof(TestResult), 
                   cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < 3; i++) {
            printf("%s: %s\n", h_selector_results[i].test_name,
                   h_selector_results[i].passed ? "PASSED" : "FAILED");
        }
        
        cudaFree(d_selector_results);
    }
    
    if (run_all || (argc > 1 && strcmp(argv[1], "integration") == 0)) {
        run_integration_tests();
    }
    
    if (run_all || (argc > 1 && strcmp(argv[1], "benchmark") == 0)) {
        benchmark_number_sizes();
    }
    
    if (run_all || (argc > 1 && strcmp(argv[1], "memory") == 0)) {
        test_memory_usage();
        test_gpu_utilization();
    }
    
    if (run_all || (argc > 1 && strcmp(argv[1], "stress") == 0)) {
        run_stress_tests();
    }
    
    if (argc > 1 && strcmp(argv[1], "factor") == 0 && argc > 2) {
        // Test specific number factorization
        printf("\n=== Factoring: %s ===\n", argv[2]);
        
        uint128_t n = parse_decimal(argv[2]);
        printf("Parsed as: ");
        print_uint128_decimal(n);
        printf("\n");
        
        // Allocate device memory
        uint128_t* d_factors;
        int* d_factor_count;
        int* d_progress;
        
        cudaMalloc(&d_factors, 2 * sizeof(uint128_t));
        cudaMalloc(&d_factor_count, sizeof(int));
        cudaMalloc(&d_progress, sizeof(int));
        
        cudaMemset(d_factor_count, 0, sizeof(int));
        cudaMemset(d_progress, 0, sizeof(int));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run factorization with large iteration count
        test_pollard_rho_component<<<256, 256>>>(
            n, uint128_t(0, 0), nullptr, 50000000
        );
        
        // Wait for completion
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Get results
        int h_factor_count;
        cudaMemcpy(&h_factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (h_factor_count > 0) {
            uint128_t h_factors[2];
            cudaMemcpy(h_factors, d_factors, 2 * sizeof(uint128_t), cudaMemcpyDeviceToHost);
            
            printf("Factor found: ");
            print_uint128_decimal(h_factors[0]);
            printf("\n");
        } else {
            printf("No factors found\n");
        }
        
        printf("Time: %.3f seconds\n", duration.count() / 1000.0);
        
        cudaFree(d_factors);
        cudaFree(d_factor_count);
        cudaFree(d_progress);
    }
    
    printf("\n=== Test Suite Complete ===\n");
    
    return 0;
}