/**
 * Test program for Quadratic Sieve implementation
 * Tests both basic and optimized versions
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include "uint128_improved.cuh"
#include "quadratic_sieve.cuh"

// Forward declarations
extern "C" bool quadratic_sieve_factor(uint128_t n, uint128_t& factor1, uint128_t& factor2);
__device__ uint32_t mod_pow(uint32_t base, uint32_t exp, uint32_t m);
__device__ uint32_t tonelli_shanks(uint64_t n, uint32_t p);
__device__ __host__ uint64_t isqrt(uint64_t n);

// Test data structure
struct TestCase {
    uint64_t n;
    uint64_t expected_factor1;
    uint64_t expected_factor2;
    const char* description;
};

// Verify factorization
bool verify_factorization(uint128_t n, uint128_t factor1, uint128_t factor2) {
    if (factor1.high != 0 || factor2.high != 0) {
        return false; // Factors too large
    }
    
    uint128_t product = uint128_t(factor1.low) * uint128_t(factor2.low);
    return product == n;
}

// Simple trial division for comparison
bool trial_division(uint64_t n, uint64_t& factor1, uint64_t& factor2) {
    if (n % 2 == 0) {
        factor1 = 2;
        factor2 = n / 2;
        return true;
    }
    
    for (uint64_t i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            factor1 = i;
            factor2 = n / i;
            return true;
        }
    }
    
    return false;
}

// Performance test function
void performance_test() {
    printf("\n=== Quadratic Sieve Performance Test ===\n");
    
    // Test cases of increasing difficulty
    std::vector<TestCase> test_cases = {
        {1009ULL * 1013ULL, 1009, 1013, "Small primes (~10 bits each)"},
        {10007ULL * 10009ULL, 10007, 10009, "Medium primes (~14 bits each)"},
        {100003ULL * 100019ULL, 100003, 100019, "Large primes (~17 bits each)"},
        {1000003ULL * 1000033ULL, 1000003, 1000033, "Very large primes (~20 bits each)"},
        {299993ULL * 314159ULL, 299993, 314159, "Different sized primes"},
        {12345678901ULL * 98765432109ULL, 12345678901ULL, 98765432109ULL, "~37 bit primes"}
    };
    
    for (const auto& test : test_cases) {
        printf("\nTest: %s\n", test.description);
        printf("n = %llu = %llu × %llu\n", test.n, test.expected_factor1, test.expected_factor2);
        
        // Time trial division
        auto start_td = std::chrono::high_resolution_clock::now();
        uint64_t td_f1, td_f2;
        bool td_success = trial_division(test.n, td_f1, td_f2);
        auto end_td = std::chrono::high_resolution_clock::now();
        auto td_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_td - start_td).count();
        
        if (td_success) {
            printf("Trial division: %llu × %llu (%.3f ms)\n", td_f1, td_f2, td_duration / 1000.0);
        }
        
        // Time Quadratic Sieve
        auto start_qs = std::chrono::high_resolution_clock::now();
        uint128_t n_128(test.n);
        uint128_t qs_f1, qs_f2;
        
        // Use optimized version
        QuadraticSieve qs(n_128);
        bool qs_success = qs.factor(qs_f1, qs_f2);
        
        auto end_qs = std::chrono::high_resolution_clock::now();
        auto qs_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_qs - start_qs).count();
        
        if (qs_success) {
            printf("Quadratic Sieve: %llu × %llu (%.3f ms)\n", qs_f1.low, qs_f2.low, qs_duration / 1000.0);
            if (verify_factorization(n_128, qs_f1, qs_f2)) {
                printf("✓ Factorization verified\n");
            } else {
                printf("✗ Factorization incorrect!\n");
            }
        } else {
            printf("Quadratic Sieve: Not implemented/failed (%.3f ms)\n", qs_duration / 1000.0);
        }
        
        // Speed comparison
        if (td_duration > 0) {
            printf("Speed ratio: %.2fx\n", (double)td_duration / qs_duration);
        }
    }
}

// Unit tests for components
void unit_tests() {
    printf("\n=== Unit Tests ===\n");
    
    // Test 1: Factor base generation
    printf("\nTest 1: Factor base generation\n");
    std::vector<QSFactorBasePrime> fb;
    uint128_t n(12345);
    qs_generate_factor_base(fb, n, 100);
    printf("Generated factor base with %zu primes\n", fb.size());
    printf("First few primes: ");
    for (size_t i = 0; i < std::min(fb.size(), size_t(10)); i++) {
        printf("%u ", fb[i].p);
    }
    printf("\n");
    
    // Test 2: Smoothness detection
    printf("\nTest 2: Smoothness detection\n");
    uint128_t smooth_num(2 * 3 * 5 * 7 * 11); // 2310
    std::vector<uint32_t> factors;
    bool is_smooth = qs_is_smooth(smooth_num, fb, factors);
    printf("Number %llu is %s\n", smooth_num.low, is_smooth ? "smooth" : "not smooth");
    if (is_smooth) {
        printf("Factors: ");
        for (uint32_t f : factors) {
            printf("%u ", fb[f].p);
        }
        printf("\n");
    }
    
    // Test 3: Polynomial generation
    printf("\nTest 3: Polynomial generation\n");
    QSPolynomial poly;
    poly.a = 1;
    poly.b = 0;
    poly.c = n;
    printf("Polynomial: Q(x) = %lldx² + %lldx - n\n", poly.a, poly.b);
    
    // Test Q(x) values
    for (int x = -5; x <= 5; x++) {
        int64_t qx = poly.a * x * x + poly.b * x;
        printf("Q(%d) = %lld - n\n", x, qx);
    }
}

// GPU kernel tests
__global__ void test_modular_arithmetic() {
    // Test modular exponentiation
    uint32_t base = 3;
    uint32_t exp = 10;
    uint32_t mod = 17;
    uint32_t result = mod_pow(base, exp, mod);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU: %u^%u mod %u = %u (expected 8)\n", base, exp, mod, result);
    }
    
    // Test Tonelli-Shanks
    uint64_t n = 10;
    uint32_t p = 13;
    uint32_t root = tonelli_shanks(n, p);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU: sqrt(%llu) mod %u = %u\n", n, p, root);
        printf("Verification: %u² mod %u = %u\n", root, p, (root * root) % p);
    }
}

int main(int argc, char* argv[]) {
    // Initialize CUDA
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Run tests based on command line arguments
    if (argc > 1 && strcmp(argv[1], "perf") == 0) {
        performance_test();
    } else if (argc > 1 && strcmp(argv[1], "gpu") == 0) {
        printf("\n=== GPU Kernel Tests ===\n");
        test_modular_arithmetic<<<1, 1>>>();
        cudaDeviceSynchronize();
    } else {
        // Run all tests
        unit_tests();
        
        printf("\n=== Basic Functionality Test ===\n");
        uint128_t test_n(299993ULL * 314159ULL);
        uint128_t f1, f2;
        
        printf("Testing with n = %llu\n", test_n.low);
        bool success = quadratic_sieve_factor(test_n, f1, f2);
        
        if (success) {
            printf("Found factors: %llu × %llu\n", f1.low, f2.low);
            if (verify_factorization(test_n, f1, f2)) {
                printf("✓ Factorization verified!\n");
            } else {
                printf("✗ Factorization incorrect!\n");
            }
        } else {
            printf("Factorization not complete (matrix solving needed)\n");
        }
    }
    
    return 0;
}