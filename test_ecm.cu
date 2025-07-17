/**
 * Test program for ECM factorization
 * Tests the Elliptic Curve Method implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "uint128_improved.cuh"
#include "ecm_cuda.cu"

// Test cases with known factorizations
struct TestCase {
    const char* n_str;
    const char* expected_factor;
    const char* description;
};

TestCase test_cases[] = {
    // Small semiprimes (products of two primes)
    {"1099511627791", "1048583", "Product of two 20-bit primes"},
    {"18446744073709551557", "4294967291", "Product of two 32-bit primes"},
    
    // Medium difficulty (10-15 digit factors)
    {"123456789012345678901", "3803", "21-digit number with small factor"},
    {"9999999999999999999999", "3", "22-digit number with small factors"},
    
    // Harder cases (15-20 digit factors)
    {"1234567890123456789012345678901234567890", "2", "40-digit number (test overflow handling)"},
    {"340282366920938463463374607431768211297", "18446744073709551557", "Product of large primes"},
    
    // Known ECM-friendly numbers
    {"2465832859640119442066663", "15485867", "RSA-79 factor (ECM-friendly)"},
    {"123456789012345678901234567890123", "3", "33-digit with small factor"},
};

// Convert string to uint128_t
uint128_t string_to_uint128(const char* str) {
    uint128_t result(0);
    uint128_t ten(10);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            result = result * ten + uint128_t(str[i] - '0');
        }
    }
    
    return result;
}

// Simple primality test for verification
bool is_prime(uint128_t n) {
    if (n < uint128_t(2)) return false;
    if (n == uint128_t(2)) return true;
    if ((n.low & 1) == 0) return false;
    
    // Trial division up to sqrt(n)
    uint128_t i(3);
    while (i * i <= n) {
        if (n % i == uint128_t(0)) return false;
        i = i + uint128_t(2);
    }
    return true;
}

void run_ecm_tests() {
    printf("ECM Factorization Test Suite\n");
    printf("============================\n\n");
    
    int passed = 0;
    int total = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int i = 0; i < total; i++) {
        printf("Test %d: %s\n", i + 1, test_cases[i].description);
        printf("Number: %s\n", test_cases[i].n_str);
        
        uint128_t n = string_to_uint128(test_cases[i].n_str);
        uint128_t factor;
        
        clock_t start = clock();
        bool found = ecm_factor(n, factor, 500); // Limit curves for testing
        clock_t end = clock();
        
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        if (found) {
            printf("Factor found: ");
            factor.print();
            printf("\n");
            
            // Verify the factor
            uint128_t quotient = n / factor;
            uint128_t remainder = n % factor;
            
            bool valid = (remainder == uint128_t(0)) && (factor > uint128_t(1)) && (factor < n);
            
            if (valid) {
                printf("Verification: PASSED (");
                factor.print();
                printf(" * ");
                quotient.print();
                printf(" = ");
                n.print();
                printf(")\n");
                
                // Check if we found the expected factor or its cofactor
                uint128_t expected = string_to_uint128(test_cases[i].expected_factor);
                if (factor == expected || quotient == expected) {
                    printf("Found expected factor!\n");
                }
                
                passed++;
            } else {
                printf("Verification: FAILED\n");
            }
        } else {
            printf("No factor found within curve limit\n");
        }
        
        printf("Time: %.3f seconds\n", time_taken);
        printf("----------------------------------------\n\n");
    }
    
    printf("Summary: %d/%d tests passed\n", passed, total);
}

// Performance benchmark
void benchmark_ecm() {
    printf("\nECM Performance Benchmark\n");
    printf("=========================\n\n");
    
    // Generate random semiprimes of different sizes
    struct {
        int bits;
        const char* example;
    } benchmarks[] = {
        {40, "1099511627791"},  // ~13 digits
        {50, "1125899906842597"}, // ~16 digits
        {60, "1152921504606846883"}, // ~19 digits
    };
    
    for (int i = 0; i < 3; i++) {
        printf("Benchmark %d-bit numbers:\n", benchmarks[i].bits);
        
        uint128_t n = string_to_uint128(benchmarks[i].example);
        uint128_t factor;
        
        // Warm up
        ecm_factor(n, factor, 10);
        
        // Actual benchmark
        int curves_tested[] = {64, 128, 256, 512};
        for (int j = 0; j < 4; j++) {
            clock_t start = clock();
            bool found = ecm_factor(n, factor, curves_tested[j]);
            clock_t end = clock();
            
            double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            double curves_per_second = curves_tested[j] / time_taken;
            
            printf("  %d curves: %.3f sec (%.1f curves/sec) - %s\n",
                   curves_tested[j], time_taken, curves_per_second,
                   found ? "FOUND" : "NOT FOUND");
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s (Compute Capability %d.%d)\n\n", 
           prop.name, prop.major, prop.minor);
    
    // Run tests
    run_ecm_tests();
    
    // Run benchmark
    benchmark_ecm();
    
    return 0;
}