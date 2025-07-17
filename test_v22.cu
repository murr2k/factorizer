#include <cstdio>
#include <cstring>
#include <chrono>

// Forward declarations
uint128_t parse_decimal(const char* str);
void print_uint128_decimal(uint128_t n);

extern "C" {
    int factorize_number(const char* number_str);
}

// Test cases
struct TestCase {
    const char* number;
    const char* expected_factors[8];
    int expected_count;
    const char* description;
};

TestCase test_cases[] = {
    // Small numbers
    {"12", {"2", "2", "3"}, 3, "Small composite"},
    {"17", {"17"}, 1, "Small prime"},
    {"100", {"2", "2", "5", "5"}, 4, "Perfect square"},
    
    // Medium numbers
    {"1234567890", {"2", "3", "3", "5", "3607", "3803"}, 6, "Medium composite"},
    {"9999999967", {"9999999967"}, 1, "Large prime"},
    
    // Large numbers
    {"123456789012345678901", {"3", "3", "7", "11", "13", "29", "101", "281", "1871", "4013"}, 10, "Large composite"},
    
    // The 26-digit challenge
    {"15482526220500967432610341", {"1804166129797", "8581541336353"}, 2, "26-digit challenge"},
    
    // Edge cases
    {"1", {}, 0, "Unity"},
    {"2", {"2"}, 1, "Smallest prime"},
    {"18446744073709551557", {"18446744073709551557"}, 1, "Large 64-bit prime"},
};

void run_test_suite() {
    printf("\n");
    printf("==================================================\n");
    printf("     CUDA Factorizer v2.2.0 - Test Suite\n");
    printf("==================================================\n\n");
    
    int passed = 0;
    int failed = 0;
    
    for (auto& test : test_cases) {
        printf("Test: %s\n", test.description);
        printf("Number: %s\n", test.number);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run factorization
        int result = factorize_number(test.number);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        printf("Time: %.3f seconds\n", duration);
        
        if (result == 0) {
            printf("Status: PASSED ✓\n");
            passed++;
        } else {
            printf("Status: FAILED ✗\n");
            failed++;
        }
        
        printf("--------------------------------------------------\n\n");
    }
    
    printf("\nTest Summary:\n");
    printf("  Total tests: %d\n", passed + failed);
    printf("  Passed: %d\n", passed);
    printf("  Failed: %d\n", failed);
    printf("  Success rate: %.1f%%\n", 100.0 * passed / (passed + failed));
    printf("==================================================\n\n");
}

int main() {
    run_test_suite();
    return 0;
}
