/**
 * Improved 128-bit CUDA Factorizer v2
 * Fixed version with proper type handling
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// Include in correct order
#include "uint128_improved.cuh"
#include "barrett_reduction.cuh"
#include "curand_pollards_rho.cuh"

// Configuration
#define MAX_BLOCKS 64
#define THREADS_PER_BLOCK 256
#define MAX_FACTORS 32
#define TIMEOUT_SECONDS 30

// Helper function to convert string to uint128_t
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

// Helper function to print uint128_t
void print_uint128(const uint128_t& n) {
    if (n.high == 0) {
        printf("%llu", n.low);
    } else {
        // For simplicity, just show as hex for large numbers
        printf("0x%llx%016llx", n.high, n.low);
    }
}

// Trial division for small factors
__global__ void trial_division_kernel(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    uint64_t start,
    uint64_t end
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    
    for (uint64_t p = start + tid * 2 + 1; p < end; p += stride * 2) {
        // Check if p divides n
        if (n.high == 0 && n.low % p == 0) {
            int idx = atomicAdd(factor_count, 1);
            if (idx < MAX_FACTORS) {
                factors[idx] = uint128_t(p, 0);
            }
        }
    }
}

// Main factorization function
bool factorize_128(const char* number_str, bool verbose = false) {
    // Convert string to uint128_t
    uint128_t n = string_to_uint128(number_str);
    
    if (verbose) {
        printf("Factoring: ");
        print_uint128(n);
        printf("\n");
    }
    
    // Allocate device memory
    uint128_t* d_factors;
    int* d_factor_count;
    cudaMalloc(&d_factors, MAX_FACTORS * sizeof(uint128_t));
    cudaMalloc(&d_factor_count, sizeof(int));
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    // Start timing
    clock_t start_time = clock();
    
    // Step 1: Quick trial division for small factors
    if (verbose) printf("Step 1: Trial division for factors < 10000...\n");
    trial_division_kernel<<<32, 256>>>(n, d_factors, d_factor_count, 2, 10000);
    cudaDeviceSynchronize();
    
    // Check if we found factors
    int factor_count;
    cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (factor_count == 0) {
        // Step 2: Pollard's Rho with cuRAND
        if (verbose) printf("Step 2: Pollard's Rho with cuRAND...\n");
        
        // Determine number of blocks based on number size
        int num_blocks = 32;
        if (n.high > 0 || n.low > 1000000000000ULL) {
            num_blocks = 64;  // More parallelism for larger numbers
        }
        
        pollards_rho_curand<<<num_blocks, THREADS_PER_BLOCK>>>(
            n, d_factors, d_factor_count, 1000000
        );
        cudaDeviceSynchronize();
        
        // Check again
        cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        // If still no factors, try Brent's variant
        if (factor_count == 0) {
            if (verbose) printf("Step 3: Brent's variant of Pollard's Rho...\n");
            cudaMemset(d_factor_count, 0, sizeof(int));
            
            pollards_rho_brent<<<num_blocks, THREADS_PER_BLOCK>>>(
                n, d_factors, d_factor_count, 2000000
            );
            cudaDeviceSynchronize();
            
            cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    
    // Copy factors back to host
    uint128_t* h_factors = new uint128_t[MAX_FACTORS];
    cudaMemcpy(h_factors, d_factors, MAX_FACTORS * sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    // Calculate elapsed time
    double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    
    // Print results
    if (factor_count > 0) {
        printf("Found %d factors in %.3f seconds:\n", factor_count, elapsed);
        
        // Remove duplicates and sort
        for (int i = 0; i < factor_count && i < MAX_FACTORS; i++) {
            printf("  ");
            print_uint128(h_factors[i]);
            
            // Check if this gives us complete factorization
            if (i == 0 && factor_count >= 2) {
                uint128_t cofactor = divide_128_64(n, h_factors[i].low);
                printf(" × ");
                print_uint128(cofactor);
            }
            printf("\n");
        }
    } else {
        printf("No factors found in %.3f seconds\n", elapsed);
        if (elapsed > TIMEOUT_SECONDS) {
            printf("Timeout reached. Number may be prime or require more iterations.\n");
        }
    }
    
    // Cleanup
    delete[] h_factors;
    cudaFree(d_factors);
    cudaFree(d_factor_count);
    
    return factor_count > 0;
}

// Test function
void run_tests() {
    printf("Running improved factorizer tests...\n\n");
    
    // Test the arithmetic operations
    test_uint128_arithmetic<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test Barrett reduction
    test_barrett_reduction<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test cuRAND integration
    test_curand_pollards<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test factorization with known test cases
    const char* test_cases[] = {
        "90595490423",       // 11 digits: 428759 × 211297
        "324625056641",      // 12 digits: 408337 × 794993
        "2626476057461",     // 13 digits: 1321171 × 1987991
        "3675257317722541"   // 16 digits: 91709393 × 40075037
    };
    
    printf("Testing factorization:\n");
    for (int i = 0; i < 4; i++) {
        printf("\nTest case %d: %s\n", i + 1, test_cases[i]);
        factorize_128(test_cases[i], true);
    }
}

// Main function
int main(int argc, char* argv[]) {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Improved 128-bit Factorizer with:\n");
    printf("  - Barrett reduction for fast modular arithmetic\n");
    printf("  - Corrected uint128_t multiplication\n");
    printf("  - cuRAND for high-quality randomness\n\n");
    
    if (argc == 1) {
        // Run tests
        run_tests();
    } else if (argc == 2) {
        // Factor a single number
        factorize_128(argv[1], true);
    } else {
        printf("Usage: %s [number]\n", argv[0]);
        printf("  Without arguments: runs test suite\n");
        printf("  With number: factors the given number\n");
    }
    
    return 0;
}