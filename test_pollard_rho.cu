/**
 * Specific test for Pollard's rho algorithm implementation
 * Tests the algorithm step by step with known values
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "uint128_improved.cuh"

// Host version of modmul for verification
uint128_t host_modmul(uint128_t a, uint128_t b, uint128_t n) {
    if (a.high == 0 && b.high == 0 && n.high == 0 && n.low != 0) {
        unsigned __int128 prod = (unsigned __int128)a.low * b.low;
        return uint128_t((uint64_t)(prod % n.low), 0);
    }
    
    // Full path
    while (a >= n) a = subtract_128(a, n);
    while (b >= n) b = subtract_128(b, n);
    
    uint256_t prod = multiply_128_128(a, b);
    uint128_t result(prod.word[0], prod.word[1]);
    
    while (result >= n) {
        result = subtract_128(result, n);
    }
    
    return result;
}

// Test kernel that runs Pollard's rho step by step
__global__ void test_pollard_step_by_step() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== Pollard's Rho Step-by-Step Test ===\n\n");
        
        // Test with n = 8051 = 83 × 97
        uint128_t n(8051, 0);
        uint128_t x(2, 0);
        uint128_t y(2, 0);
        uint128_t c(1, 0);
        
        printf("Testing factorization of 8051 = 83 × 97\n");
        printf("Starting with x = y = 2, c = 1\n\n");
        
        for (int i = 0; i < 20; i++) {
            // Single step for x
            x = modmul_128_fast(x, x, n);
            x = add_128(x, c);
            if (x >= n) x = subtract_128(x, n);
            
            // Double step for y
            y = modmul_128_fast(y, y, n);
            y = add_128(y, c);
            if (y >= n) y = subtract_128(y, n);
            
            y = modmul_128_fast(y, y, n);
            y = add_128(y, c);
            if (y >= n) y = subtract_128(y, n);
            
            // Calculate difference
            uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
            
            // Calculate GCD
            uint128_t g = gcd_128(diff, n);
            
            printf("Iteration %2d: x=%4llu, y=%4llu, |x-y|=%4llu, gcd=%llu\n",
                   i+1, x.low, y.low, diff.low, g.low);
            
            if (g.low > 1 && g < n) {
                printf("\nFactor found: %llu\n", g.low);
                printf("Cofactor: %llu\n", n.low / g.low);
                printf("Verification: %llu × %llu = %llu\n", 
                       g.low, n.low / g.low, g.low * (n.low / g.low));
                break;
            }
        }
        
        printf("\n=== Testing with larger number ===\n");
        
        // Test with n = 1000000007 × 1000000009
        uint128_t n2(1000000016000000063ULL, 0);
        x = uint128_t(2, 0);
        y = uint128_t(2, 0);
        c = uint128_t(1, 0);
        
        printf("\nTesting factorization of 1000000016000000063 = 1000000007 × 1000000009\n");
        printf("First 10 iterations:\n");
        
        for (int i = 0; i < 10; i++) {
            // Single step for x
            x = modmul_128_fast(x, x, n2);
            x = add_128(x, c);
            if (x >= n2) x = subtract_128(x, n2);
            
            // Double step for y
            y = modmul_128_fast(y, y, n2);
            y = add_128(y, c);
            if (y >= n2) y = subtract_128(y, n2);
            
            y = modmul_128_fast(y, y, n2);
            y = add_128(y, c);
            if (y >= n2) y = subtract_128(y, n2);
            
            // Calculate difference
            uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
            
            // Calculate GCD
            uint128_t g = gcd_128(diff, n2);
            
            printf("Iteration %2d: x=%llu, y=%llu, gcd=%llu\n",
                   i+1, x.low, y.low, g.low);
            
            if (g.low > 1 && g < n2) {
                printf("\nFactor found: %llu\n", g.low);
                break;
            }
        }
    }
}

// Test the modular multiplication specifically
__global__ void test_modmul_specific() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Testing Modular Multiplication ===\n");
        
        // Test case 1: Small numbers
        uint128_t a(5, 0);
        uint128_t b(7, 0);
        uint128_t n(13, 0);
        uint128_t result = modmul_128_fast(a, b, n);
        printf("5 × 7 mod 13 = %llu (expected: 9)\n", result.low);
        
        // Test case 2: Squaring
        uint128_t x(12345, 0);
        uint128_t n2(1000000, 0);
        result = modmul_128_fast(x, x, n2);
        printf("12345² mod 1000000 = %llu (expected: %llu)\n", 
               result.low, (12345ULL * 12345ULL) % 1000000);
        
        // Test case 3: Large modulus
        uint128_t large_n(1000000016000000063ULL, 0);
        uint128_t val(1000000000ULL, 0);
        result = modmul_128_fast(val, val, large_n);
        printf("10⁹ × 10⁹ mod 1000000016000000063 = %llu\n", result.low);
        
        // Test the sequence x = x² + 1 mod n
        printf("\n=== Testing x = x² + 1 sequence ===\n");
        x = uint128_t(2, 0);
        n = uint128_t(8051, 0);
        uint128_t c(1, 0);
        
        printf("n = 8051, starting with x = 2:\n");
        for (int i = 0; i < 10; i++) {
            x = modmul_128_fast(x, x, n);
            x = add_128(x, c);
            if (x >= n) x = subtract_128(x, n);
            printf("  x[%d] = %llu\n", i+1, x.low);
        }
    }
}

// Test GCD function with various inputs
__global__ void test_gcd_specific() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Testing GCD Function ===\n");
        
        // Test cases
        struct {
            uint64_t a, b, expected;
        } tests[] = {
            {48, 18, 6},
            {1000000007, 1000000009, 1},
            {83*97, 83*89, 83},
            {2*3*5*7, 3*5*11, 15},
            {0, 15, 15},
            {15, 0, 15},
            {1, 1000000, 1}
        };
        
        for (int i = 0; i < 7; i++) {
            uint128_t a(tests[i].a, 0);
            uint128_t b(tests[i].b, 0);
            uint128_t result = gcd_128(a, b);
            printf("gcd(%llu, %llu) = %llu (expected: %llu) %s\n",
                   tests[i].a, tests[i].b, result.low, tests[i].expected,
                   result.low == tests[i].expected ? "✓" : "✗");
        }
    }
}

int main() {
    printf("=== Pollard's Rho Algorithm Detailed Test ===\n\n");
    
    // Run tests
    test_modmul_specific<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    test_gcd_specific<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    test_pollard_step_by_step<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n=== Tests Complete ===\n");
    return 0;
}