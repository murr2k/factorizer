/**
 * Test program for 128-bit arithmetic operations
 * Verifies correctness of basic operations
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "uint128_improved.cuh"

// Test modular multiplication implementations
__global__ void test_modmul_implementations() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== Testing Modular Multiplication ===\n\n");
        
        // Test 1: Small numbers
        uint128_t a(12345, 0);
        uint128_t b(67890, 0);
        uint128_t n(1000000, 0);
        
        // Original method
        uint128_t result1 = a;
        if (result1 >= n) {
            while (result1 >= n) {
                result1 = subtract_128(result1, n);
            }
        }
        uint128_t result2 = b;
        if (result2 >= n) {
            while (result2 >= n) {
                result2 = subtract_128(result2, n);
            }
        }
        uint256_t prod = multiply_128_128(result1, result2);
        uint128_t result_orig(prod.word[0], prod.word[1]);
        while (result_orig >= n) {
            result_orig = subtract_128(result_orig, n);
        }
        
        printf("Test 1: %llu * %llu mod %llu\n", a.low, b.low, n.low);
        printf("  Expected: %llu\n", (12345ULL * 67890ULL) % 1000000ULL);
        printf("  Original method: %llu\n", result_orig.low);
        
        // Test 2: Large 64-bit numbers
        uint128_t a2(0xFFFFFFFFFFFFFFFFULL, 0);
        uint128_t b2(2, 0);
        uint128_t n2(0xFFFFFFFFFFFFFFFFULL, 0);
        
        printf("\nTest 2: (2^64-1) * 2 mod (2^64-1)\n");
        printf("  a = 0x%llx:%llx\n", a2.high, a2.low);
        printf("  b = %llu\n", b2.low);
        printf("  n = 0x%llx:%llx\n", n2.high, n2.low);
        
        // The result should be 0 because (2^64-1) * 2 = 2^65 - 2 = 2*(2^64-1)
        // So mod (2^64-1) = 0
        
        // Test 3: 128-bit numbers
        uint128_t a3(0x123456789ABCDEF0ULL, 0x1);
        uint128_t b3(0x2, 0);
        uint128_t n3(0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL);
        
        printf("\nTest 3: Large 128-bit multiplication\n");
        printf("  a = 0x%llx:%llx\n", a3.high, a3.low);
        printf("  b = %llu\n", b3.low);
        printf("  n = 0x%llx:%llx\n", n3.high, n3.low);
        
        uint256_t prod3 = multiply_128_128(a3, b3);
        printf("  a * b = 0x%llx:%llx:%llx:%llx\n", 
               prod3.word[3], prod3.word[2], prod3.word[1], prod3.word[0]);
    }
}

// Test GCD implementation
__global__ void test_gcd() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Testing GCD ===\n\n");
        
        // Test 1: Simple case
        uint128_t a(48, 0);
        uint128_t b(18, 0);
        uint128_t result = gcd_128(a, b);
        printf("Test 1: gcd(48, 18) = %llu (expected: 6)\n", result.low);
        
        // Test 2: One zero
        uint128_t c(0, 0);
        uint128_t d(15, 0);
        result = gcd_128(c, d);
        printf("Test 2: gcd(0, 15) = %llu (expected: 15)\n", result.low);
        
        // Test 3: Large numbers
        uint128_t e(0x123456789ABCDEF0ULL, 0);
        uint128_t f(0x369D036A0ULL, 0);  // This is e / 6
        result = gcd_128(e, f);
        printf("Test 3: gcd of large numbers = 0x%llx:%llx\n", result.high, result.low);
        
        // Test 4: Coprime numbers
        uint128_t g(1234567, 0);
        uint128_t h(7654321, 0);
        result = gcd_128(g, h);
        printf("Test 4: gcd(1234567, 7654321) = %llu (expected: 1)\n", result.low);
    }
}

// Test arithmetic overflow cases
__global__ void test_overflow_cases() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Testing Overflow Cases ===\n\n");
        
        // Test 1: Addition overflow
        uint128_t max_low(0xFFFFFFFFFFFFFFFFULL, 0);
        uint128_t one(1, 0);
        uint128_t sum = add_128(max_low, one);
        printf("Test 1: 0xFFFFFFFFFFFFFFFF + 1 = 0x%llx:%llx (expected: 0x1:0x0)\n", 
               sum.high, sum.low);
        
        // Test 2: Subtraction underflow
        uint128_t zero(0, 0);
        uint128_t sub = subtract_128(zero, one);
        printf("Test 2: 0 - 1 = 0x%llx:%llx (expected: 0xFFFFFFFFFFFFFFFF:0xFFFFFFFFFFFFFFFF)\n",
               sub.high, sub.low);
        
        // Test 3: Multiplication overflow
        uint128_t big(0xFFFFFFFFFFFFFFFFULL, 0);
        uint256_t prod = multiply_128_128(big, big);
        printf("Test 3: (2^64-1)^2 = 0x%llx:%llx:%llx:%llx\n",
               prod.word[3], prod.word[2], prod.word[1], prod.word[0]);
        printf("        Expected: 0x0:0x0:0xFFFFFFFFFFFFFFFE:0x0000000000000001\n");
    }
}

// Test Pollard's rho specific cases
__global__ void test_pollard_cases() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Testing Pollard's Rho Cases ===\n\n");
        
        // Test with a known composite number
        uint128_t n(299, 0);  // 13 * 23
        uint128_t x(2, 0);
        uint128_t c(1, 0);
        
        printf("Testing with n = 299 = 13 × 23\n");
        
        for (int i = 0; i < 20; i++) {
            // x = (x^2 + c) mod n
            uint128_t x_squared = x;
            if (x_squared >= n) {
                while (x_squared >= n) {
                    x_squared = subtract_128(x_squared, n);
                }
            }
            uint256_t prod = multiply_128_128(x_squared, x_squared);
            x = uint128_t(prod.word[0], prod.word[1]);
            while (x >= n) {
                x = subtract_128(x, n);
            }
            x = add_128(x, c);
            if (x >= n) {
                x = subtract_128(x, n);
            }
            
            printf("  Iteration %d: x = %llu\n", i, x.low);
            
            if (i > 5) break;  // Just show first few iterations
        }
    }
}

int main() {
    printf("=== 128-bit Arithmetic Test Suite ===\n\n");
    
    // Launch test kernels
    test_modmul_implementations<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    test_gcd<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    test_overflow_cases<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    test_pollard_cases<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Also run the built-in test
    printf("\n=== Running Built-in Tests ===\n");
    test_uint128_arithmetic<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Tests Complete ===\n");
    
    return 0;
}