/**
 * Small number test for v2.2.0 implementation
 * Tests factorization with progressively larger numbers
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"

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

// Simple Pollard's Rho kernel for testing
__global__ void pollards_rho_simple(
    uint128_t n,
    uint128_t* factor,
    int* found
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return; // Single thread for simplicity
    
    // Setup Barrett reduction
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // Initialize
    uint128_t x(2, 0);
    uint128_t y(2, 0);
    uint128_t c(1, 0);
    
    // Pollard's f function
    auto f = [&](const uint128_t& val) {
        uint256_t squared = multiply_128_128(val, val);
        uint128_t result = barrett.reduce(squared);
        result = add_128(result, c);
        if (result >= n) {
            result = subtract_128(result, n);
        }
        return result;
    };
    
    // Main loop
    for (int i = 0; i < 1000000 && !(*found); i++) {
        x = f(x);
        y = f(f(y));
        
        uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
        uint128_t g = gcd_128(diff, n);
        
        if (g > uint128_t(1, 0) && g < n) {
            *factor = g;
            *found = 1;
            return;
        }
    }
}

void test_number(const char* num_str, const char* description) {
    printf("\n=== Testing: %s ===\n", description);
    printf("Number: %s\n", num_str);
    
    uint128_t n = parse_decimal(num_str);
    printf("Binary: %016llx%016llx\n", n.high, n.low);
    
    // Device memory
    uint128_t* d_factor;
    int* d_found;
    cudaMalloc(&d_factor, sizeof(uint128_t));
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    pollards_rho_simple<<<1, 1>>>(n, d_factor, d_found);
    cudaDeviceSynchronize();
    
    // Get results
    int h_found;
    uint128_t h_factor;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_factor, d_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    if (h_found) {
        printf("Factor found: ");
        print_uint128_decimal(h_factor);
        
        // Calculate cofactor
        uint256_t n_256;
        n_256.word[0] = n.low;
        n_256.word[1] = n.high;
        n_256.word[2] = 0;
        n_256.word[3] = 0;
        uint128_t cofactor = divide_256_128(n_256, h_factor);
        
        printf(" × ");
        print_uint128_decimal(cofactor);
        printf("\n");
    } else {
        printf("No factor found\n");
    }
    
    printf("Time: %.3f seconds\n", elapsed);
    
    // Cleanup
    cudaFree(d_factor);
    cudaFree(d_found);
}

int main() {
    printf("CUDA Factorizer v2.2.0 - Small Number Tests\n");
    printf("============================================\n");
    
    // Test cases with increasing difficulty
    test_number("77", "Small composite (7×11)");
    test_number("1001", "Medium composite (7×11×13)");
    test_number("12345", "5-digit number");
    test_number("1234567", "7-digit number");
    test_number("123456789", "9-digit number");
    test_number("47703785443", "11-digit number from earlier test");
    test_number("918399205110619", "15-digit number from earlier test");
    
    printf("\n============================================\n");
    printf("Tests completed.\n");
    
    return 0;
}