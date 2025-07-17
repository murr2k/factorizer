/**
 * Optimized factorization for 86-bit number: 71123818302723020625487649
 * Based on v2.1 simple approach
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include <unistd.h>

// Parse decimal string properly
uint128_t parse_decimal_correct(const char* str) {
    uint128_t result(0, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            // Multiply by 10
            uint128_t temp = result;
            result = add_128(result, result); // 2x
            result = add_128(result, result); // 4x
            result = add_128(result, temp);   // 5x
            result = add_128(result, result); // 10x
            
            // Add digit
            result = add_128(result, uint128_t(str[i] - '0', 0));
        }
    }
    
    return result;
}

// Print decimal properly
void print_decimal(uint128_t n) {
    if (n.high == 0 && n.low == 0) {
        printf("0");
        return;
    }
    
    char digits[40];
    int pos = 0;
    
    uint128_t ten(10, 0);
    while (!n.is_zero()) {
        // n mod 10
        uint128_t quotient(0, 0);
        uint64_t remainder = 0;
        
        // Simple division by 10
        uint128_t temp = n;
        int count = 0;
        while (temp >= ten) {
            temp = subtract_128(temp, ten);
            count++;
        }
        quotient = uint128_t(count, 0);
        remainder = temp.low;
        
        digits[pos++] = '0' + remainder;
        n = quotient;
    }
    
    // Print in reverse
    for (int i = pos - 1; i >= 0; i--) {
        printf("%c", digits[i]);
    }
}

// Optimized Pollard's Rho 
__global__ void pollards_rho_optimized(
    uint128_t n,
    uint128_t* factor,
    int* found,
    int max_iter = 100000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Setup PRNG
    curandState_t state;
    curand_init(clock64() + tid * 31337, tid, 0, &state);
    
    // Barrett reduction setup
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // Start near sqrt(n) ≈ 8433493837237
    uint64_t sqrt_estimate = 8433493837237ULL;
    uint64_t range = sqrt_estimate / 100; // 1% range
    
    uint128_t x(sqrt_estimate - range + (curand(&state) % (2 * range)), 0);
    uint128_t y = x;
    uint128_t c(1 + (curand(&state) % 100), 0);
    
    // Brent's optimization
    int m = 100;
    uint128_t ys = y;
    uint128_t product(1, 0);
    
    // Pollard's f
    auto f = [&](const uint128_t& val) {
        uint256_t squared = multiply_128_128(val, val);
        uint128_t result = barrett.reduce(squared);
        result = add_128(result, c);
        if (result >= n) {
            result = subtract_128(result, n);
        }
        return result;
    };
    
    for (int i = 0; i < max_iter && !(*found); i++) {
        // Brent's algorithm
        if (i % m == 0) {
            ys = y;
            product = uint128_t(1, 0);
        }
        
        y = f(y);
        
        uint128_t diff = (y > ys) ? subtract_128(y, ys) : subtract_128(ys, y);
        uint256_t prod = multiply_128_128(product, diff);
        product = barrett.reduce(prod);
        
        if (i % m == m - 1) {
            uint128_t g = gcd_128(product, n);
            
            if (g > uint128_t(1, 0) && g < n) {
                *factor = g;
                atomicExch(found, 1);
                return;
            }
        }
        
        // Restart with new parameters
        if (i % 1000000 == 999999) {
            x = uint128_t(sqrt_estimate - range + (curand(&state) % (2 * range)), 0);
            y = x;
            c = uint128_t(1 + (curand(&state) % 1000), 0);
            ys = y;
            product = uint128_t(1, 0);
        }
    }
}

int main() {
    printf("=== Optimized 86-bit Factorization ===\n");
    const char* num_str = "71123818302723020625487649";
    printf("Number: %s\n", num_str);
    
    // Parse correctly
    uint128_t n = parse_decimal_correct(num_str);
    
    // Verify parsing
    printf("Parsed: ");
    print_decimal(n);
    printf("\n");
    printf("Hex: %016llx%016llx\n", n.high, n.low);
    printf("Bits: %d\n\n", 128 - n.leading_zeros());
    
    // GPU memory
    uint128_t* d_factor;
    int* d_found;
    cudaMalloc(&d_factor, sizeof(uint128_t));
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));
    
    // Launch
    int blocks = 128;
    int threads = 256;
    printf("Launching %d blocks × %d threads\n", blocks, threads);
    printf("Starting near sqrt(n) ≈ 8433493837237\n\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    pollards_rho_optimized<<<blocks, threads>>>(n, d_factor, d_found);
    
    // Poll for results
    int h_found = 0;
    int seconds = 0;
    
    while (seconds < 60 && !h_found) {
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (h_found) break;
        
        if (seconds % 10 == 0) {
            printf("Searching... %d seconds\n", seconds);
        }
        
        sleep(1);
        seconds++;
    }
    
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Results
    uint128_t h_factor;
    cudaMemcpy(&h_factor, d_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    printf("\nTime: %.3f seconds\n", elapsed);
    
    if (h_found && !h_factor.is_zero()) {
        printf("\n✓ Factor found: ");
        print_decimal(h_factor);
        
        // Compute cofactor
        uint256_t n_256;
        n_256.word[0] = n.low;
        n_256.word[1] = n.high;
        n_256.word[2] = 0;
        n_256.word[3] = 0;
        uint128_t cofactor = divide_256_128(n_256, h_factor);
        
        printf("\n  Cofactor: ");
        print_decimal(cofactor);
        printf("\n");
        
        // Verify
        uint256_t check = multiply_128_128(h_factor, cofactor);
        if (uint128_t(check.word[0], check.word[1]) == n) {
            printf("\n✓ Verified!\n");
        }
    } else {
        printf("\n✗ No factors found in %d seconds\n", seconds);
        printf("This may be a prime or have very large factors.\n");
        printf("Consider using ECM or Quadratic Sieve.\n");
    }
    
    cudaFree(d_factor);
    cudaFree(d_found);
    
    return 0;
}