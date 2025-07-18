/**
 * CUDA Factorizer v2.3.2 - Hybrid Smart Edition
 * Uses multiple strategies based on number characteristics
 * 
 * Features:
 * - Trial division for small factors
 * - Fermat's method for factors near sqrt(n)
 * - GPU-accelerated parallel search
 * - Smart bounds based on bit size
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

// Include core components
#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"

// Version information
#define VERSION_MAJOR 2
#define VERSION_MINOR 3
#define VERSION_PATCH 2
#define VERSION_STRING "2.3.2-Hybrid"

// Configuration
#define MAX_TRIAL_DIVISION 1000000
#define GPU_BLOCKS 512
#define GPU_THREADS 256

// Forward declarations
void print_uint128_decimal(uint128_t n);
uint128_t parse_decimal(const char* str);
uint64_t isqrt(uint128_t n);

// Simple trial division for small factors
bool trial_division(uint128_t n, uint128_t& factor1, uint128_t& factor2, uint64_t limit) {
    if (n.high == 0) {
        // For 64-bit numbers, use simple division
        uint64_t num = n.low;
        for (uint64_t d = 2; d <= limit && d * d <= num; d++) {
            if (num % d == 0) {
                factor1 = uint128_t(d, 0);
                factor2 = uint128_t(num / d, 0);
                return true;
            }
        }
    }
    return false;
}

// Fermat's factorization method - good for factors near sqrt(n)
bool fermat_factorization(uint128_t n, uint128_t& factor1, uint128_t& factor2) {
    if ((n.low & 1) == 0) return false; // Even numbers handled by trial division
    
    // Start with a = ceil(sqrt(n))
    uint64_t sqrt_n = isqrt(n);
    uint128_t a(sqrt_n, 0);
    uint256_t a_squared = multiply_128_128(a, a);
    uint128_t a_squared_128(a_squared.word[0], a_squared.word[1]);
    if (a_squared_128 < n) {
        a = add_128(a, uint128_t(1, 0));
    }
    
    // Try up to 10 million iterations
    for (int i = 0; i < 10000000; i++) {
        uint256_t a2 = multiply_128_128(a, a);
        uint128_t a2_mod(a2.word[0], a2.word[1]);
        
        if (a2_mod >= n) {
            uint128_t b2 = subtract_128(a2_mod, n);
            
            // Check if b2 is a perfect square
            uint64_t b = isqrt(b2);
            if (b2.high == 0 && b * b == b2.low) {
                // Found factors: n = (a-b)(a+b)
                uint128_t b128(b, 0);
                factor1 = subtract_128(a, b128);
                factor2 = add_128(a, b128);
                return true;
            }
        }
        
        a = add_128(a, uint128_t(1, 0));
    }
    
    return false;
}

// GPU kernel for parallel factor search
__global__ void factor_search_kernel(
    uint128_t n,
    uint64_t start_value,
    uint64_t range_per_thread,
    uint128_t* factor,
    int* found
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread searches a range
    uint64_t my_start = start_value + tid * range_per_thread;
    uint64_t my_end = my_start + range_per_thread;
    
    // Ensure we don't go beyond sqrt(n)
    uint64_t sqrt_n_approx = 1ULL << ((128 - n.leading_zeros()) / 2);
    if (my_end > sqrt_n_approx) my_end = sqrt_n_approx;
    
    // Search the range
    for (uint64_t d = my_start; d < my_end && !(*found); d++) {
        if (d < 2) continue;
        
        // Check if d divides n
        if (n.high == 0 && n.low % d == 0) {
            *factor = uint128_t(d, 0);
            atomicExch(found, 1);
            return;
        }
        
        // For larger numbers, use modular arithmetic
        if (n.high > 0) {
            // Simplified check for divisibility
            uint128_t remainder = n;
            uint128_t divisor(d, 0);
            
            // Subtract divisor repeatedly (simplified modulo)
            while (remainder >= divisor) {
                remainder = subtract_128(remainder, divisor);
                if (remainder.is_zero()) {
                    *factor = divisor;
                    atomicExch(found, 1);
                    return;
                }
                
                // Early exit if remainder is getting too small
                if (remainder < divisor) break;
            }
        }
    }
}

// Smart factorization using multiple methods
bool smart_factor(uint128_t n, uint128_t& factor1, uint128_t& factor2) {
    printf("Starting smart factorization...\n");
    
    int bit_size = 128 - n.leading_zeros();
    printf("Number has %d bits\n", bit_size);
    
    // Step 1: Quick trial division for small factors
    printf("Step 1: Checking small factors up to %d...\n", MAX_TRIAL_DIVISION);
    if (trial_division(n, factor1, factor2, MAX_TRIAL_DIVISION)) {
        printf("Found via trial division!\n");
        return true;
    }
    
    // Step 2: Try Fermat's method (good for factors near sqrt(n))
    printf("Step 2: Trying Fermat's method...\n");
    if (fermat_factorization(n, factor1, factor2)) {
        printf("Found via Fermat's method!\n");
        return true;
    }
    
    // Step 3: GPU-accelerated search
    printf("Step 3: GPU-accelerated factor search...\n");
    
    // Allocate device memory
    uint128_t* d_factor;
    int* d_found;
    cudaMalloc(&d_factor, sizeof(uint128_t));
    cudaMalloc(&d_found, sizeof(int));
    
    // Initialize
    uint128_t h_factor(0, 0);
    int h_found = 0;
    cudaMemcpy(d_factor, &h_factor, sizeof(uint128_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate search parameters
    uint64_t sqrt_n_approx = isqrt(n);
    uint64_t total_threads = GPU_BLOCKS * GPU_THREADS;
    uint64_t search_start = MAX_TRIAL_DIVISION + 1;
    uint64_t search_range = sqrt_n_approx - search_start;
    uint64_t range_per_thread = (search_range + total_threads - 1) / total_threads;
    
    printf("GPU searching from %llu to ~%llu with %d threads\n", 
           (unsigned long long)search_start, 
           (unsigned long long)sqrt_n_approx, 
           total_threads);
    
    // Launch kernel
    factor_search_kernel<<<GPU_BLOCKS, GPU_THREADS>>>(
        n, search_start, range_per_thread, d_factor, d_found
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Check result
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found) {
        cudaMemcpy(&h_factor, d_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
        factor1 = h_factor;
        
        // Calculate factor2
        if (n.high == 0 && h_factor.high == 0) {
            factor2 = uint128_t(n.low / h_factor.low, 0);
        } else {
            // Use Barrett division
            uint256_t n_256;
            n_256.word[0] = n.low;
            n_256.word[1] = n.high;
            n_256.word[2] = 0;
            n_256.word[3] = 0;
            factor2 = divide_256_128(n_256, h_factor);
        }
        
        cudaFree(d_factor);
        cudaFree(d_found);
        printf("Found via GPU search!\n");
        return true;
    }
    
    cudaFree(d_factor);
    cudaFree(d_found);
    
    // Step 4: If we reach here, we need more sophisticated methods
    printf("Basic methods failed. Number likely has large balanced factors.\n");
    printf("Would need full ECM or QS implementation.\n");
    
    return false;
}

// Integer square root
uint64_t isqrt(uint128_t n) {
    if (n.high == 0) {
        return (uint64_t)sqrt((double)n.low);
    }
    
    // For large numbers, use Newton's method
    uint64_t x = 1ULL << ((128 - n.leading_zeros()) / 2);
    uint64_t prev = 0;
    
    while (x != prev) {
        prev = x;
        // x = (x + n/x) / 2
        if (n.high == 0) {
            x = (x + n.low / x) / 2;
        } else {
            // Approximation for very large numbers
            x = (x + (n.high / x)) / 2;
        }
    }
    
    return x;
}

// Parse decimal string to uint128
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

// Print uint128 in decimal
void print_uint128_decimal(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", (unsigned long long)n.low);
        return;
    }
    
    char buffer[40];
    int pos = 39;
    buffer[pos] = '\0';
    
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

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        printf("Example: %s 139789207152250802634791\n", argv[0]);
        return 1;
    }
    
    // Parse number
    uint128_t n = parse_decimal(argv[1]);
    
    // Check CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("Error: No CUDA-capable devices found\n");
        return 1;
    }
    
    cudaSetDevice(0);
    
    printf("CUDA Factorizer v%s - Hybrid Smart Edition\n", VERSION_STRING);
    printf("Target number: ");
    print_uint128_decimal(n);
    printf("\n\n");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run factorization
    uint128_t factor1, factor2;
    bool success = smart_factor(n, factor1, factor2);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end_time - start_time).count();
    
    if (success) {
        printf("\n✓ Factorization successful!\n");
        printf("Time: %.3f seconds\n", elapsed);
        printf("Factor 1: ");
        print_uint128_decimal(factor1);
        printf(" (%d bits)\n", 128 - factor1.leading_zeros());
        printf("Factor 2: ");
        print_uint128_decimal(factor2);
        printf(" (%d bits)\n", 128 - factor2.leading_zeros());
        
        // Verify
        uint256_t check = multiply_128_128(factor1, factor2);
        if (uint128_t(check.word[0], check.word[1]) == n) {
            printf("✓ Verification: factors multiply to original number\n");
        }
        
        return 0;
    } else {
        printf("\n✗ Factorization failed\n");
        printf("Time: %.3f seconds\n", elapsed);
        return 1;
    }
}