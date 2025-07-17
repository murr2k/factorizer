/**
 * Minimal 128-bit Factorizer
 * Testing without cuRAND to isolate issues
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "uint128_improved.cuh"
#include "barrett_reduction.cuh"

#define MAX_FACTORS 32

// Simple Pollard's Rho without cuRAND
__global__ void pollards_rho_simple(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    int max_iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple deterministic start based on thread ID
    uint128_t x((tid + 2) % n.low, 0);
    uint128_t y = x;
    uint128_t c(1 + (tid % 10), 0);
    
    // Barrett reduction setup
    Barrett128 barrett;
    barrett.n = n;
    barrett.precompute();
    
    uint128_t factor(1, 0);
    
    for (int i = 0; i < max_iterations && factor.low == 1; i++) {
        // x = (x^2 + c) mod n
        uint256_t x_squared = multiply_128_128(x, x);
        x = barrett.reduce(x_squared.low_128());
        x = add_128(x, c);
        if (x >= n) x = subtract_128(x, n);
        
        // y = f(f(y))
        for (int j = 0; j < 2; j++) {
            uint256_t y_squared = multiply_128_128(y, y);
            y = barrett.reduce(y_squared.low_128());
            y = add_128(y, c);
            if (y >= n) y = subtract_128(y, n);
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
            break;
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
    
    // Run Pollard's Rho
    pollards_rho_simple<<<32, 256>>>(n, d_factors, d_factor_count, 100000);
    cudaDeviceSynchronize();
    
    // Get results
    int factor_count;
    cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (factor_count > 0) {
        uint128_t* h_factors = new uint128_t[MAX_FACTORS];
        cudaMemcpy(h_factors, d_factors, MAX_FACTORS * sizeof(uint128_t), cudaMemcpyDeviceToHost);
        
        printf("Found %d factor(s) in %.3f seconds:\n", 
               factor_count, (double)(clock() - start) / CLOCKS_PER_SEC);
        
        for (int i = 0; i < factor_count && i < MAX_FACTORS; i++) {
            printf("  %llu\n", h_factors[i].low);
        }
        
        // Verify
        if (factor_count >= 1) {
            uint128_t cofactor = divide_128_64(n, h_factors[0].low);
            printf("Factorization: %llu × %llu = %s\n", 
                   h_factors[0].low, cofactor.low, argv[1]);
        }
        
        delete[] h_factors;
    } else {
        printf("No factors found in %.3f seconds\n", 
               (double)(clock() - start) / CLOCKS_PER_SEC);
    }
    
    // Cleanup
    cudaFree(d_factors);
    cudaFree(d_factor_count);
    
    return 0;
}