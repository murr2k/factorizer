/**
 * Simple test for 26-digit factorization
 * Using known factors to test the approach
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "uint128_improved.cuh"

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
    
    // Simple conversion for display
    printf("0x%016llx%016llx", n.high, n.low);
}

__global__ void verify_factors(uint128_t n, uint128_t factor1, uint128_t factor2, int* valid) {
    // Simply verify the factorization
    uint256_t product = multiply_128_128(factor1, factor2);
    uint128_t result(product.word[0], product.word[1]);
    
    if (result == n) {
        *valid = 1;
    } else {
        *valid = 0;
    }
}

int main() {
    printf("26-Digit Number Factorization Test\n");
    printf("==================================\n\n");
    
    // The 26-digit challenge number
    const char* number_str = "15482526220500967432610341";
    uint128_t n = parse_decimal(number_str);
    
    // Known factors
    const char* factor1_str = "1804166129797";
    const char* factor2_str = "8581541336353";
    uint128_t factor1 = parse_decimal(factor1_str);
    uint128_t factor2 = parse_decimal(factor2_str);
    
    printf("Number: %s\n", number_str);
    printf("Factor 1: %s\n", factor1_str);
    printf("Factor 2: %s\n", factor2_str);
    
    // Test on GPU
    int* d_valid;
    cudaMalloc(&d_valid, sizeof(int));
    
    verify_factors<<<1, 1>>>(n, factor1, factor2, d_valid);
    cudaDeviceSynchronize();
    
    int h_valid;
    cudaMemcpy(&h_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_valid) {
        printf("\n✓ Factorization verified on GPU!\n");
        printf("  %s × %s = %s\n", factor1_str, factor2_str, number_str);
    } else {
        printf("\n✗ Factorization failed!\n");
    }
    
    // Test basic arithmetic
    printf("\nTesting 128-bit arithmetic:\n");
    printf("Factor 1 (hex): ");
    print_uint128_decimal(factor1);
    printf("\n");
    printf("Factor 2 (hex): ");
    print_uint128_decimal(factor2);
    printf("\n");
    printf("Product (hex): ");
    print_uint128_decimal(n);
    printf("\n");
    
    // Calculate bit sizes
    printf("\nBit sizes:\n");
    printf("Factor 1: %d bits\n", 128 - factor1.leading_zeros());
    printf("Factor 2: %d bits\n", 128 - factor2.leading_zeros());  
    printf("Product: %d bits\n", 128 - n.leading_zeros());
    
    cudaFree(d_valid);
    
    printf("\n==================================\n");
    printf("Conclusion: The 26-digit number has two ~43-bit prime factors.\n");
    printf("This is challenging for Pollard's Rho but feasible with enough time.\n");
    printf("For faster results, consider Quadratic Sieve or ECM.\n");
    
    return 0;
}