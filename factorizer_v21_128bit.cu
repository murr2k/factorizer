/**
 * v2.1 Factorizer for 128-bit numbers
 * Handles up to 39 decimal digits
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include "uint128_improved.cuh"

// Parse decimal string to uint128
uint128_t parse_decimal(const char* str) {
    uint128_t result(0, 0);
    uint128_t ten(10, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            // result = result * 10 + digit
            uint256_t temp = multiply_128_128(result, ten);
            result = uint128_t(temp.word[0], temp.word[1]);
            result = add_128(result, uint128_t(str[i] - '0', 0));
        }
    }
    
    return result;
}

// Print uint128 in decimal
void print_uint128(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
        return;
    }
    
    // For larger numbers, just show hex for now
    printf("0x%llx%016llx", n.high, n.low);
}

// Optimized modular multiplication for 128-bit
__device__ uint128_t modmul_128(uint128_t a, uint128_t b, uint128_t n) {
    // Simple method: ensure a and b are reduced first
    if (a >= n) {
        // Reduce a mod n (simplified)
        while (a >= n) {
            a = subtract_128(a, n);
        }
    }
    if (b >= n) {
        while (b >= n) {
            b = subtract_128(b, n);
        }
    }
    
    // Multiply and reduce
    uint256_t prod = multiply_128_128(a, b);
    uint128_t result(prod.word[0], prod.word[1]);
    
    // Simple reduction
    while (result >= n) {
        result = subtract_128(result, n);
    }
    
    return result;
}

__global__ void pollards_rho_128bit(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    int max_iterations = 50000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize variables
    uint128_t x(2 + tid, 0);
    uint128_t y = x;
    uint128_t c(1 + (tid % 100), 0);
    uint128_t factor(1, 0);
    
    for (int i = 0; i < max_iterations; i++) {
        // x = (x^2 + c) mod n
        x = modmul_128(x, x, n);
        x = add_128(x, c);
        if (x >= n) x = subtract_128(x, n);
        
        // y = f(f(y))
        y = modmul_128(y, y, n);
        y = add_128(y, c);
        if (y >= n) y = subtract_128(y, n);
        
        y = modmul_128(y, y, n);
        y = add_128(y, c);
        if (y >= n) y = subtract_128(y, n);
        
        // Calculate |x - y|
        uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
        
        // Calculate GCD
        factor = gcd_128(diff, n);
        
        if (factor.low > 1 && factor < n) {
            // Found a factor!
            int idx = atomicAdd(factor_count, 1);
            if (idx == 0) {
                factors[0] = factor;
                // Try to get cofactor
                if (n.high == 0 && factor.high == 0 && factor.low != 0) {
                    factors[1] = uint128_t(n.low / factor.low, 0);
                }
            }
            break;
        }
        
        // Warp cooperation
        if (i % 10000 == 0) {
            unsigned mask = __ballot_sync(0xFFFFFFFF, factor.low > 1);
            if (mask != 0) break;
        }
        
        // Re-randomize periodically
        if (i % (100000 + tid * 1000) == 0) {
            c = uint128_t((c.low + tid + i) % 100 + 1, 0);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }
    
    // Parse input
    uint128_t n = parse_decimal(argv[1]);
    
    printf("=== v2.1 128-bit Factorizer ===\n");
    printf("Input: %s\n", argv[1]);
    printf("Number: ");
    print_uint128(n);
    printf("\n");
    
    // Check if it's odd (for Montgomery)
    printf("Optimization: %s\n\n", (n.low & 1) ? "Montgomery-capable" : "Standard");
    
    // Allocate device memory
    uint128_t* d_factors;
    int* d_factor_count;
    cudaMalloc(&d_factors, 2 * sizeof(uint128_t));
    cudaMalloc(&d_factor_count, sizeof(int));
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    // Configure grid - more threads for larger numbers
    int blocks = 128;
    int threads = 256;
    printf("Launching %d blocks x %d threads = %d total threads\n", 
           blocks, threads, blocks * threads);
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    pollards_rho_128bit<<<blocks, threads>>>(n, d_factors, d_factor_count);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Get timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get results
    int h_factor_count;
    uint128_t h_factors[2];
    cudaMemcpy(&h_factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_factors, d_factors, 2 * sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    printf("\nTime: %.3f seconds\n", duration.count() / 1000.0);
    
    if (h_factor_count > 0 && h_factors[0].low > 1) {
        printf("✓ Factors found:\n");
        printf("  Factor 1: ");
        print_uint128(h_factors[0]);
        printf("\n");
        
        if (h_factors[1].low > 1) {
            printf("  Factor 2: ");
            print_uint128(h_factors[1]);
            printf("\n");
            
            // Verify if both are 64-bit
            if (h_factors[0].high == 0 && h_factors[1].high == 0) {
                printf("\nVerification: %llu × %llu = %llu\n", 
                       h_factors[0].low, h_factors[1].low, 
                       h_factors[0].low * h_factors[1].low);
            }
        }
    } else {
        printf("✗ No factors found\n");
    }
    
    // Cleanup
    cudaFree(d_factors);
    cudaFree(d_factor_count);
    
    return h_factor_count > 0 ? 0 : 1;
}