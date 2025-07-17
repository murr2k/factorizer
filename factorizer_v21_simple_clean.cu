/**
 * Factorizer v2.1 Simple Clean - Pollard's Rho with simpler modular arithmetic
 * Target: 15482526220500967432610341 (26 digits)
 * Known factors: 1804166129797 × 8581541336353
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <unistd.h>
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
void print_uint128_decimal(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
        return;
    }
    
    // Convert to string by repeated division
    char buffer[40];
    int pos = 39;
    buffer[pos] = '\0';
    
    while (!n.is_zero() && pos > 0) {
        // Division by 10 using simple method
        uint128_t quotient(0, 0);
        uint64_t remainder = 0;
        
        // Process high word
        if (n.high > 0) {
            remainder = n.high % 10;
            quotient.high = n.high / 10;
        }
        
        // Process low word with carry
        unsigned __int128 temp = (unsigned __int128)remainder << 64 | n.low;
        quotient.low = temp / 10;
        remainder = temp % 10;
        
        buffer[--pos] = '0' + remainder;
        n = quotient;
    }
    
    printf("%s", &buffer[pos]);
}

// Simpler modular reduction for 256-bit to 128-bit
__device__ uint128_t mod_reduce(const uint256_t& x, const uint128_t& n) {
    // Start with the low 128 bits
    uint128_t result(x.word[0], x.word[1]);
    
    // If high part is zero, just reduce the result
    if (x.word[2] == 0 && x.word[3] == 0) {
        while (result >= n) {
            result = subtract_128(result, n);
        }
        return result;
    }
    
    // Binary reduction method
    uint128_t high_part(x.word[2], x.word[3]);
    
    // Process bit by bit from the high part
    for (int i = 0; i < 128; i++) {
        // Shift result left by 1
        uint64_t carry = (result.high >> 63) & 1;
        result.high = (result.high << 1) | (result.low >> 63);
        result.low = result.low << 1;
        
        // Add bit from high part
        if (high_part.high & (1ULL << 63)) {
            result.low |= 1;
        }
        
        // Shift high part left
        high_part.high = (high_part.high << 1) | (high_part.low >> 63);
        high_part.low = high_part.low << 1;
        
        // Reduce if needed
        if (result >= n) {
            result = subtract_128(result, n);
        }
    }
    
    return result;
}

// Pollard's Rho function: x_{n+1} = x_n^2 + c (mod n)
__device__ uint128_t pollard_f(const uint128_t& x, const uint128_t& c, const uint128_t& n) {
    // x^2 mod n
    uint256_t x_squared = multiply_128_128(x, x);
    uint128_t result = mod_reduce(x_squared, n);
    
    // Add c
    result = add_128(result, c);
    if (result >= n) {
        result = subtract_128(result, n);
    }
    
    return result;
}

// Pollard's Rho kernel - single thread version
__global__ void pollard_rho_single(
    uint128_t n,
    uint128_t* device_factor,
    volatile int* device_status,
    int max_iterations
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    printf("GPU: Starting Pollard's Rho\n");
    printf("GPU: n = %llx:%llx\n", n.high, n.low);
    
    // Initialize with fixed values
    uint128_t x(2, 0);
    uint128_t y(2, 0);
    uint128_t c(1, 0);
    
    int iterations = 0;
    int progress_interval = 10000;
    
    while (iterations < max_iterations && *device_status == 0) {
        // Floyd's cycle detection
        x = pollard_f(x, c, n);
        y = pollard_f(pollard_f(y, c, n), c, n);
        
        // Calculate |x - y|
        uint128_t diff;
        if (x >= y) {
            diff = subtract_128(x, y);
        } else {
            diff = subtract_128(y, x);
        }
        
        // Calculate GCD
        uint128_t d = gcd_128(diff, n);
        
        iterations++;
        
        // Progress report
        if (iterations % progress_interval == 0) {
            printf("GPU: Iteration %d, diff=%llx:%llx, gcd=%llx:%llx\n", 
                   iterations, diff.high, diff.low, d.high, d.low);
        }
        
        // Check if we found a non-trivial factor
        if (!d.is_zero() && d != uint128_t(1, 0) && d != n) {
            printf("GPU: Found potential factor at iteration %d!\n", iterations);
            printf("GPU: Factor d = %llx:%llx\n", d.high, d.low);
            
            *device_factor = d;
            *device_status = 1;
            return;
        }
        
        // Change parameters if stuck
        if (iterations % 100000 == 0 && iterations > 0) {
            c = add_128(c, uint128_t(1, 0));
            x = uint128_t(2, 0);
            y = uint128_t(2, 0);
            printf("GPU: Changing c to %llu at iteration %d\n", c.low, iterations);
        }
    }
    
    printf("GPU: No factor found after %d iterations\n", iterations);
}

// Test division function
__global__ void test_division() {
    // Test with known values
    uint256_t n_ext;
    n_ext.word[0] = 0x6d9df611b42b6225ULL;  // Low part of target
    n_ext.word[1] = 0xcce8dULL;              // High part of target
    n_ext.word[2] = 0;
    n_ext.word[3] = 0;
    
    // Known factor: 1804166129797 = 0x1a44e6f0485
    uint128_t factor(0x1a44e6f0485ULL, 0);
    
    printf("GPU Test: Attempting division of target by known factor\n");
    printf("Target: %llx:%llx\n", n_ext.word[1], n_ext.word[0]);
    printf("Factor: %llx:%llx\n", factor.high, factor.low);
    
    // We need a proper division implementation
    // For now, let's just verify multiplication works
    uint128_t cofactor(0x7cce96e28e1ULL, 0);  // Other known factor
    uint256_t product = multiply_128_128(factor, cofactor);
    
    printf("Product of factors: %llx:%llx:%llx:%llx\n",
           product.word[3], product.word[2], product.word[1], product.word[0]);
    printf("Should match target: %llx:%llx\n", n_ext.word[1], n_ext.word[0]);
}

int main(int argc, char* argv[]) {
    // Target number: 15482526220500967432610341
    const char* target = "15482526220500967432610341";
    
    printf("CUDA Factorizer v2.1 Simple Clean\n");
    printf("Target: %s\n", target);
    
    // Parse the target number
    uint128_t n = parse_decimal(target);
    printf("Parsed as uint128: high=%llx, low=%llx\n", n.high, n.low);
    
    // Verify parsing
    printf("Decimal representation: ");
    print_uint128_decimal(n);
    printf("\n\n");
    
    // First run a division test
    printf("Running division test...\n");
    test_division<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Allocate device memory
    uint128_t* device_factor;
    int* device_status;
    cudaMalloc(&device_factor, sizeof(uint128_t));
    cudaMalloc(&device_status, sizeof(int));
    cudaMemset(device_status, 0, sizeof(int));
    
    // Run Pollard's Rho
    printf("Running Pollard's Rho...\n");
    pollard_rho_single<<<1, 1>>>(n, device_factor, device_status, 1000000);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Check result
    int status;
    uint128_t factor;
    cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&factor, device_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    if (status == 1) {
        printf("\nFactor found: ");
        print_uint128_decimal(factor);
        printf(" (hex: %llx:%llx)\n", factor.high, factor.low);
        
        // We would verify here but need proper division
        printf("\nTo verify: %s = ", target);
        print_uint128_decimal(factor);
        printf(" × ???\n");
    } else {
        printf("\nNo factor found in initial run.\n");
    }
    
    // Cleanup
    cudaFree(device_factor);
    cudaFree(device_status);
    
    return 0;
}