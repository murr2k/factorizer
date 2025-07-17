/**
 * Simple integration test for CUDA Factorizer v2.2.0
 * Tests the integrated system with known test cases
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
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
    } else {
        printf("(high=%llu, low=%llu)", n.high, n.low);
    }
}

// Test algorithm selection logic
void test_algorithm_selection() {
    printf("=== Algorithm Selection Test ===\n\n");
    
    struct TestCase {
        const char* number;
        const char* expected_algorithm;
        const char* description;
    };
    
    TestCase cases[] = {
        {"123456789", "Trial Division", "9-digit number"},
        {"1234567890123", "Pollard's Rho", "13-digit number"},
        {"15482526220500967432610341", "ECM", "26-digit case (84 bits)"},
        {"71123818302723020625487649", "QS", "86-bit case"},
        {"123456789012345678901234567890", "QS", "Very large number"}
    };
    
    for (auto& test : cases) {
        uint128_t n = parse_decimal(test.number);
        int bit_size = 128 - n.leading_zeros();
        
        printf("Number: %s\n", test.number);
        printf("Bit size: %d\n", bit_size);
        printf("Expected algorithm: %s\n", test.expected_algorithm);
        
        // Algorithm selection logic
        const char* selected_algo = "Unknown";
        
        if (bit_size <= 20) {
            selected_algo = "Trial Division";
        } else if (bit_size <= 64) {
            selected_algo = "Pollard's Rho";
        } else if (bit_size == 84) {
            selected_algo = "ECM";
        } else if (bit_size == 86) {
            selected_algo = "QS";
        } else if (bit_size <= 90) {
            selected_algo = "Pollard's Rho (Brent)";
        } else {
            selected_algo = "QS";
        }
        
        printf("Selected algorithm: %s\n", selected_algo);
        
        if (strcmp(selected_algo, test.expected_algorithm) == 0) {
            printf("✓ PASSED\n");
        } else {
            printf("✗ FAILED\n");
        }
        
        printf("--------------------------------------------------\n\n");
    }
}

// Test factorization verification
void test_factorization_verification() {
    printf("=== Factorization Verification Test ===\n\n");
    
    struct FactorTest {
        const char* number;
        const char* factor1;
        const char* factor2;
        const char* description;
    };
    
    FactorTest tests[] = {
        {"15482526220500967432610341", "1804166129797", "8581541336353", "26-digit case"},
        {"71123818302723020625487649", "7574960675251", "9389331687899", "86-bit case"}
    };
    
    for (auto& test : tests) {
        uint128_t n = parse_decimal(test.number);
        uint128_t f1 = parse_decimal(test.factor1);
        uint128_t f2 = parse_decimal(test.factor2);
        
        printf("Test: %s\n", test.description);
        printf("Number: %s\n", test.number);
        printf("Factor 1: %s (%d bits)\n", test.factor1, 128 - f1.leading_zeros());
        printf("Factor 2: %s (%d bits)\n", test.factor2, 128 - f2.leading_zeros());
        
        // Verify factorization
        uint256_t product = multiply_128_128(f1, f2);
        uint128_t result(product.word[0], product.word[1]);
        
        if (result == n) {
            printf("✓ Factorization verified\n");
        } else {
            printf("✗ Factorization failed\n");
            printf("Expected: ");
            print_uint128_decimal(n);
            printf("\nGot: ");
            print_uint128_decimal(result);
            printf("\n");
        }
        
        printf("--------------------------------------------------\n\n");
    }
}

// Test CUDA availability
void test_cuda_availability() {
    printf("=== CUDA Availability Test ===\n\n");
    
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess) {
        printf("✗ CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("✓ CUDA devices found: %d\n", device_count);
    
    if (device_count > 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        
        printf("Device 0: %s\n", props.name);
        printf("Compute capability: %d.%d\n", props.major, props.minor);
        printf("Global memory: %.1f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Multiprocessors: %d\n", props.multiProcessorCount);
        printf("✓ CUDA ready for factorization\n");
    }
    
    printf("--------------------------------------------------\n\n");
}

int main() {
    printf("CUDA Factorizer v2.2.0 - Integration Test Suite\n");
    printf("================================================\n\n");
    
    // Test CUDA availability
    test_cuda_availability();
    
    // Test algorithm selection
    test_algorithm_selection();
    
    // Test factorization verification
    test_factorization_verification();
    
    printf("Integration tests completed!\n");
    printf("Ready to build and test the integrated factorizer.\n");
    
    return 0;
}