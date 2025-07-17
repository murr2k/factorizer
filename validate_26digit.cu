/**
 * Quick validation program for 26-digit number factorization
 * Tests: 15482526220500967432610341 = 1804166129797 × 8581541336353
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main() {
    printf("=== 26-Digit Number Validation ===\n\n");
    
    // The target number
    const char* target = "15482526220500967432610341";
    const char* factor1_str = "1804166129797";
    const char* factor2_str = "8581541336353";
    
    // Parse all numbers
    uint128_t n = parse_decimal(target);
    uint128_t f1 = parse_decimal(factor1_str);
    uint128_t f2 = parse_decimal(factor2_str);
    
    printf("Target number: %s\n", target);
    printf("Expected factors: %s × %s\n\n", factor1_str, factor2_str);
    
    // Verify the factorization
    printf("Verification:\n");
    printf("Factor 1: ");
    print_uint128_decimal(f1);
    printf(" (0x%llx%016llx)\n", f1.high, f1.low);
    
    printf("Factor 2: ");
    print_uint128_decimal(f2);
    printf(" (0x%llx%016llx)\n", f2.high, f2.low);
    
    // Multiply factors
    uint256_t product = multiply_128_128(f1, f2);
    
    printf("\nProduct (256-bit):\n");
    printf("  Word[3]: 0x%016llx\n", product.word[3]);
    printf("  Word[2]: 0x%016llx\n", product.word[2]);
    printf("  Word[1]: 0x%016llx\n", product.word[1]);
    printf("  Word[0]: 0x%016llx\n", product.word[0]);
    
    // Check if product matches original
    bool matches = (product.word[0] == n.low && 
                   product.word[1] == n.high && 
                   product.word[2] == 0 && 
                   product.word[3] == 0);
    
    printf("\nProduct as uint128: ");
    uint128_t prod_128(product.word[0], product.word[1]);
    print_uint128_decimal(prod_128);
    printf("\n");
    
    printf("\nVerification: %s\n", matches ? "✓ PASSED" : "✗ FAILED");
    
    if (matches) {
        printf("The factorization is correct!\n");
        
        // Additional analysis
        printf("\nFactor Analysis:\n");
        printf("  Factor 1 bits: %d\n", 128 - f1.leading_zeros());
        printf("  Factor 2 bits: %d\n", 128 - f2.leading_zeros());
        printf("  Product bits: %d\n", 128 - n.leading_zeros());
        
        // Check if factors are prime (basic test)
        printf("\nPrimality hints:\n");
        printf("  Factor 1 (%s): ", factor1_str);
        if ((f1.low & 1) == 0) printf("Even (not prime)\n");
        else if (f1.low % 3 == 0) printf("Divisible by 3 (not prime)\n");
        else if (f1.low % 5 == 0) printf("Divisible by 5 (not prime)\n");
        else printf("Possibly prime (needs full test)\n");
        
        printf("  Factor 2 (%s): ", factor2_str);
        if ((f2.low & 1) == 0) printf("Even (not prime)\n");
        else if (f2.low % 3 == 0) printf("Divisible by 3 (not prime)\n");
        else if (f2.low % 5 == 0) printf("Divisible by 5 (not prime)\n");
        else printf("Possibly prime (needs full test)\n");
    }
    
    return matches ? 0 : 1;
}