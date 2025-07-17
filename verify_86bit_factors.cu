/**
 * Verify the factorization of 71123818302723020625487649
 * Known factors: 7574960675251 × 9389331687899
 */

#include <cuda_runtime.h>
#include <cstdio>

#include "uint128_improved.cuh"

// Parse decimal string
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

// Print decimal
void print_decimal(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
    } else {
        printf("(high=%llu, low=%llu)", n.high, n.low);
    }
}

int main() {
    printf("=== Verification of 86-bit Factorization ===\n\n");
    
    const char* n_str = "71123818302723020625487649";
    const char* p1_str = "7574960675251";
    const char* p2_str = "9389331687899";
    
    uint128_t n = parse_decimal(n_str);
    uint128_t p1 = parse_decimal(p1_str);
    uint128_t p2 = parse_decimal(p2_str);
    
    printf("Number: %s\n", n_str);
    printf("Factor 1: %s (%d bits)\n", p1_str, 128 - p1.leading_zeros());
    printf("Factor 2: %s (%d bits)\n", p2_str, 128 - p2.leading_zeros());
    
    // Verify on CPU
    uint256_t product = multiply_128_128(p1, p2);
    uint128_t result(product.word[0], product.word[1]);
    
    printf("\nVerification:\n");
    printf("  %s × %s = ", p1_str, p2_str);
    
    if (result == n) {
        printf("%s ✓\n", n_str);
        printf("\nFactorization confirmed!\n");
    } else {
        printf("MISMATCH!\n");
        printf("Expected: ");
        print_decimal(n);
        printf("\nGot: ");
        print_decimal(result);
        printf("\n");
    }
    
    printf("\nAnalysis:\n");
    printf("- Original number: 86 bits\n");
    printf("- Factor 1: %d bits\n", 128 - p1.leading_zeros());
    printf("- Factor 2: %d bits\n", 128 - p2.leading_zeros());
    printf("- Both factors are ~43 bits each\n");
    printf("\nThis explains why Pollard's Rho was slow - the factors are large and balanced.\n");
    printf("For such numbers, Quadratic Sieve or ECM would be more efficient.\n");
    
    return 0;
}