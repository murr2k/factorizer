/**
 * Working 128-bit Factorizer
 * Demonstrates the improvements with validated test cases
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

#include "uint128_improved.cuh"

#define MAX_FACTORS 32
#define THREADS_PER_BLOCK 256

// Simple modulo for small numbers
__device__ uint128_t simple_mod(const uint128_t& a, const uint128_t& n) {
    if (n.high == 0 && a.high == 0) {
        return uint128_t(a.low % n.low, 0);
    }
    
    uint128_t result = a;
    while (result >= n) {
        result = subtract_128(result, n);
    }
    return result;
}

// Pollard's Rho kernel
__global__ void pollards_rho_kernel(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    int max_iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Multiple starting points based on thread ID
    uint128_t x((2 + tid * 3) % 1000, 0);
    uint128_t y = x;
    uint128_t c(1 + (tid % 10), 0);
    
    uint128_t factor(1, 0);
    
    for (int i = 0; i < max_iterations && factor.low == 1; i++) {
        // x = (x^2 + c) mod n
        if (n.high == 0) {
            // Optimize for small n
            uint64_t x_val = (x.low * x.low + c.low) % n.low;
            x = uint128_t(x_val, 0);
        } else {
            uint256_t x_squared = multiply_128_128(x, x);
            x = simple_mod(x_squared.low_128(), n);
            x = add_128(x, c);
            x = simple_mod(x, n);
        }
        
        // y = f(f(y))
        for (int j = 0; j < 2; j++) {
            if (n.high == 0) {
                uint64_t y_val = (y.low * y.low + c.low) % n.low;
                y = uint128_t(y_val, 0);
            } else {
                uint256_t y_squared = multiply_128_128(y, y);
                y = simple_mod(y_squared.low_128(), n);
                y = add_128(y, c);
                y = simple_mod(y, n);
            }
        }
        
        // Calculate GCD
        uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
        factor = gcd_128(diff, n);
        
        // Check if found non-trivial factor
        if (factor.low > 1 && factor < n) {
            int idx = atomicAdd(factor_count, 1);
            if (idx < MAX_FACTORS) {
                factors[idx] = factor;
            }
            return;
        }
    }
}

// Helper to convert string to uint128_t
uint128_t string_to_uint128(const char* str) {
    uint128_t result(0, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        // result = result * 10 + digit
        uint128_t ten(10, 0);
        uint256_t prod = multiply_128_128(result, ten);
        result = prod.low_128();
        result = add_128(result, uint128_t(str[i] - '0', 0));
    }
    
    return result;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number>\n", argv[0]);
        printf("\nTest with validated cases:\n");
        printf("  %s 90595490423        # 11-digit: 428759 × 211297\n", argv[0]);
        printf("  %s 324625056641       # 12-digit: 408337 × 794993\n", argv[0]);
        printf("  %s 2626476057461      # 13-digit: 1321171 × 1987991\n", argv[0]);
        return 1;
    }
    
    // Convert input
    uint128_t n = string_to_uint128(argv[1]);
    printf("Factoring: %s\n", argv[1]);
    
    // Allocate device memory
    uint128_t* d_factors;
    int* d_factor_count;
    cudaMalloc(&d_factors, MAX_FACTORS * sizeof(uint128_t));
    cudaMalloc(&d_factor_count, sizeof(int));
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    // Start timing
    clock_t start = clock();
    
    // Run Pollard's Rho with multiple blocks/threads
    int num_blocks = 32;
    pollards_rho_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        n, d_factors, d_factor_count, 1000000
    );
    cudaDeviceSynchronize();
    
    // Get results
    int factor_count;
    cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    if (factor_count > 0) {
        uint128_t* h_factors = new uint128_t[MAX_FACTORS];
        cudaMemcpy(h_factors, d_factors, MAX_FACTORS * sizeof(uint128_t), cudaMemcpyDeviceToHost);
        
        printf("✓ Found %d factor(s) in %.3f seconds:\n", factor_count, elapsed);
        
        // Show unique factors
        for (int i = 0; i < factor_count && i < MAX_FACTORS; i++) {
            printf("  Factor: %llu\n", h_factors[i].low);
            
            // Calculate cofactor using simple division for verification
            if (n.high == 0 && h_factors[i].high == 0 && h_factors[i].low != 0) {
                uint64_t cofactor = n.low / h_factors[i].low;
                if (h_factors[i].low * cofactor == n.low) {
                    printf("  Verification: %llu × %llu = %s ✓\n", 
                           h_factors[i].low, cofactor, argv[1]);
                }
            }
        }
        
        delete[] h_factors;
    } else {
        printf("✗ No factors found in %.3f seconds\n", elapsed);
        printf("  (Number may be prime or require more iterations)\n");
    }
    
    // Cleanup
    cudaFree(d_factors);
    cudaFree(d_factor_count);
    
    return 0;
}