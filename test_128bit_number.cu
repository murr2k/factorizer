/**
 * Test factorization of: 71123818302723020625487649
 * Using proper 128-bit arithmetic
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include <unistd.h>

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

// Pollard's f function
__device__ uint128_t pollards_f(const uint128_t& x, const uint128_t& c, const Barrett128_v2& barrett) {
    uint256_t x_squared = multiply_128_128(x, x);
    uint128_t result = barrett.reduce(x_squared);
    
    result = add_128(result, c);
    if (result >= barrett.n) {
        result = subtract_128(result, barrett.n);
    }
    
    return result;
}

// Pollard's Rho kernel
__global__ void pollards_rho_kernel(
    uint128_t n,
    uint128_t* factor,
    int* found,
    int max_iterations = 50000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize PRNG
    curandState_t state;
    curand_init(clock64() + tid * 7919, tid, 0, &state);
    
    // Setup Barrett reduction
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // Initialize values
    uint128_t x(2 + curand(&state) % 1000000, 0);
    uint128_t y = x;
    uint128_t c(1 + curand(&state) % 1000, 0);
    
    // Brent's variant
    int m = 128;
    int r = 1;
    uint128_t ys = y;
    uint128_t product(1, 0);
    
    for (int i = 0; i < max_iterations && !(*found); i++) {
        if (i == r) {
            ys = y;
            r *= 2;
        }
        
        // Batch GCD computation
        for (int j = 0; j < m && i + j < r; j++) {
            y = pollards_f(y, c, barrett);
            
            uint128_t diff = (y > ys) ? subtract_128(y, ys) : subtract_128(ys, y);
            uint256_t prod = multiply_128_128(product, diff);
            product = barrett.reduce(prod);
        }
        
        // Check GCD
        uint128_t g = gcd_128(product, n);
        
        if (g > uint128_t(1, 0) && g < n) {
            *factor = g;
            atomicExch(found, 1);
            return;
        }
        
        // Reset product periodically
        if (i % 1000 == 0) {
            product = uint128_t(1, 0);
        }
        
        // Change parameters occasionally
        if (i % (100000 + tid * 1000) == 0) {
            c = uint128_t(1 + curand(&state) % 100000, 0);
            x = uint128_t(2 + curand(&state) % 1000000000ULL, 0);
            y = x;
            ys = y;
            product = uint128_t(1, 0);
        }
    }
}

int main() {
    printf("=== 128-bit Factorization Test ===\n");
    printf("Number: 71123818302723020625487649\n\n");
    
    // Parse the number
    const char* number_str = "71123818302723020625487649";
    uint128_t n = parse_decimal(number_str);
    
    printf("Parsed: ");
    print_uint128_decimal(n);
    printf("\n");
    printf("Binary: %016llx%016llx\n", n.high, n.low);
    printf("Bit size: %d\n\n", 128 - n.leading_zeros());
    
    // Allocate device memory
    uint128_t* d_factor;
    int* d_found;
    cudaMalloc(&d_factor, sizeof(uint128_t));
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));
    
    // Launch configuration
    int num_blocks = 64;
    int threads_per_block = 256;
    int total_threads = num_blocks * threads_per_block;
    
    printf("Launching %d threads (%d blocks × %d threads/block)\n", 
           total_threads, num_blocks, threads_per_block);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    pollards_rho_kernel<<<num_blocks, threads_per_block>>>(n, d_factor, d_found);
    
    // Wait for completion (with timeout)
    int seconds = 0;
    int h_found = 0;
    
    while (seconds < 30 && !h_found) {
        if (cudaStreamQuery(0) == cudaSuccess) {
            break;
        }
        
        // Check if found
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (seconds % 5 == 0) {
            printf("Working... %d seconds\n", seconds);
        }
        
        sleep(1);
        seconds++;
    }
    
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Get results
    uint128_t h_factor;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_factor, d_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    printf("\nTime: %.3f seconds\n", elapsed);
    
    if (h_found && h_factor.low > 1) {
        printf("✓ Factor found: ");
        print_uint128_decimal(h_factor);
        printf("\n");
        
        // Calculate cofactor
        uint256_t n_256;
        n_256.word[0] = n.low;
        n_256.word[1] = n.high;
        n_256.word[2] = 0;
        n_256.word[3] = 0;
        uint128_t cofactor = divide_256_128(n_256, h_factor);
        
        printf("  Cofactor: ");
        print_uint128_decimal(cofactor);
        printf("\n\n");
        
        // Verify
        uint256_t product = multiply_128_128(h_factor, cofactor);
        uint128_t check(product.word[0], product.word[1]);
        
        if (check == n) {
            printf("✓ Factorization verified!\n");
        }
        
        // Show bit sizes
        printf("\nFactor bit sizes:\n");
        printf("  Factor 1: %d bits\n", 128 - h_factor.leading_zeros());
        printf("  Factor 2: %d bits\n", 128 - cofactor.leading_zeros());
    } else {
        printf("✗ No factors found within time limit\n");
        printf("The number may be prime or have large factors.\n");
    }
    
    // Cleanup
    cudaFree(d_factor);
    cudaFree(d_found);
    
    return 0;
}