/**
 * Quick factorization test for: 71123818302723020625487649
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Simple trial division on CPU first
void trial_division(unsigned long long n, int max_trial = 1000000) {
    printf("Trial division up to %d...\n", max_trial);
    
    // Check small primes
    if (n % 2 == 0) {
        printf("Factor found: 2\n");
        return;
    }
    
    for (int d = 3; d <= max_trial && d <= sqrt(n); d += 2) {
        if (n % d == 0) {
            printf("Factor found: %d\n", d);
            printf("Cofactor: %llu\n", n / d);
            return;
        }
    }
    
    printf("No small factors found up to %d\n", max_trial);
}

// Check if the string represents a number that fits in 64 bits
bool fits_in_64bit(const char* str) {
    // Max 64-bit unsigned: 18446744073709551615 (20 digits)
    int len = strlen(str);
    if (len > 20) return false;
    if (len < 20) return true;
    
    // If 20 digits, compare with max value
    return strcmp(str, "18446744073709551615") <= 0;
}

int main() {
    const char* number_str = "71123818302723020625487649";
    
    printf("=== Quick Factorization Test ===\n");
    printf("Number: %s\n", number_str);
    printf("Length: %d digits\n\n", (int)strlen(number_str));
    
    // Check if it fits in 64 bits
    if (!fits_in_64bit(number_str)) {
        printf("Number is larger than 64 bits (>18446744073709551615)\n");
        printf("This is a %d-digit number requiring 128-bit arithmetic.\n\n", (int)strlen(number_str));
        
        // Check last digit for divisibility hints
        int last_digit = number_str[strlen(number_str)-1] - '0';
        printf("Last digit: %d\n", last_digit);
        
        if (last_digit % 2 == 0) {
            printf("Number is even (divisible by 2)\n");
        } else if (last_digit == 5) {
            printf("Number is divisible by 5\n");
        } else {
            printf("Number is odd and not divisible by 5\n");
            
            // Sum of digits for divisibility by 3
            int digit_sum = 0;
            for (int i = 0; number_str[i]; i++) {
                digit_sum += (number_str[i] - '0');
            }
            printf("Sum of digits: %d\n", digit_sum);
            if (digit_sum % 3 == 0) {
                printf("Number is divisible by 3\n");
            } else if (digit_sum % 9 == 0) {
                printf("Number is divisible by 9\n");
            } else {
                printf("Number is not divisible by 3 or 9\n");
            }
        }
        
        // Try small primes
        printf("\nChecking small prime divisibility...\n");
        
        // We can check the number modulo small primes
        // by working with the last few digits
        unsigned long long last_10_digits = 0;
        int start = strlen(number_str) - 10;
        if (start < 0) start = 0;
        for (int i = start; number_str[i]; i++) {
            last_10_digits = last_10_digits * 10 + (number_str[i] - '0');
        }
        
        printf("Last 10 digits: %llu\n", last_10_digits);
        
        // Quick checks
        int small_primes[] = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
        for (int i = 0; i < 12; i++) {
            if (last_10_digits % small_primes[i] == 0) {
                printf("Last 10 digits divisible by %d (full number might be too)\n", 
                       small_primes[i]);
            }
        }
        
    } else {
        // Parse as 64-bit
        unsigned long long n = 0;
        for (int i = 0; number_str[i]; i++) {
            n = n * 10 + (number_str[i] - '0');
        }
        
        printf("Number fits in 64 bits: %llu\n\n", n);
        
        // Try trial division
        trial_division(n);
    }
    
    printf("\nFor full factorization of this 26-digit number,\n");
    printf("we need 128-bit arithmetic and advanced algorithms.\n");
    
    return 0;
}