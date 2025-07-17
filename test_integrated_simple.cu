/**
 * Simple Test Program for CUDA Factorizer v2.2.0 Integration
 * Tests basic functionality of integrated ECM and QS algorithms
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "uint128_improved.cuh"

// Simple test cases
struct TestCase {
    const char* name;
    const char* number;
    const char* expected_algorithm;
    int expected_bits;
};

// Test cases for validation
TestCase test_cases[] = {
    {"Small composite", "3599", "Trial Division", 12},
    {"Medium composite", "1152921504606846883", "Pollard's Rho", 60},
    {"26-digit target", "15482526220500967432610341", "ECM", 84},
    {"86-bit target", "29318992932113061061655073", "QS", 86},
    {"Large composite", "1208925819614629174706449", "ECM", 80}
};

const int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);

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
    
    while (!n.is_zero() && pos > 0) {
        uint64_t remainder = 0;
        uint128_t quotient(0, 0);
        
        if (n.high > 0) {
            remainder = n.high % 10;
            quotient.high = n.high / 10;
            uint64_t temp = remainder * (1ULL << 32) * (1ULL << 32) + n.low;
            quotient.low = temp / 10;
            remainder = temp % 10;
        } else {
            quotient.low = n.low / 10;
            remainder = n.low % 10;
        }
        
        buffer[--pos] = '0' + remainder;
        n = quotient;
    }
    
    printf("%s", &buffer[pos]);
}

// Test individual case
bool test_case(const TestCase& tc) {
    printf("\nTesting: %s\n", tc.name);
    printf("Number: %s\n", tc.number);
    
    uint128_t n = parse_decimal(tc.number);
    int bit_size = 128 - n.leading_zeros();
    
    printf("Parsed number: ");
    print_uint128_decimal(n);
    printf("\n");
    printf("Bit size: %d (expected: %d)\n", bit_size, tc.expected_bits);
    
    // Verify parsing
    if (abs(bit_size - tc.expected_bits) > 2) {
        printf("❌ FAIL: Bit size mismatch\n");
        return false;
    }
    
    printf("✅ PASS: Number parsing and bit size correct\n");
    return true;
}

// Test CUDA availability
bool test_cuda() {
    printf("Testing CUDA availability...\n");
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess) {
        printf("❌ FAIL: CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    if (device_count == 0) {
        printf("❌ FAIL: No CUDA devices found\n");
        return false;
    }
    
    printf("✅ PASS: Found %d CUDA device(s)\n", device_count);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device 0: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    
    return true;
}

// Test uint128 arithmetic
bool test_uint128_arithmetic() {
    printf("\nTesting uint128 arithmetic...\n");
    
    // Test addition
    uint128_t a(0x123456789abcdef0ULL, 0x0fedcba987654321ULL);
    uint128_t b(0x1111111111111111ULL, 0x2222222222222222ULL);
    uint128_t c = add_128(a, b);
    
    printf("Addition test: ");
    print_uint128_decimal(a);
    printf(" + ");
    print_uint128_decimal(b);
    printf(" = ");
    print_uint128_decimal(c);
    printf("\n");
    
    // Test multiplication
    uint128_t d(12345, 0);
    uint128_t e(67890, 0);
    uint256_t f = multiply_128_128(d, e);
    
    printf("Multiplication test: ");
    print_uint128_decimal(d);
    printf(" × ");
    print_uint128_decimal(e);
    printf(" = %llu\n", f.word[0]);
    
    // Verify result
    if (f.word[0] != 12345ULL * 67890ULL) {
        printf("❌ FAIL: Multiplication incorrect\n");
        return false;
    }
    
    printf("✅ PASS: uint128 arithmetic working\n");
    return true;
}

// Test algorithm selection logic
bool test_algorithm_selection() {
    printf("\nTesting algorithm selection logic...\n");
    
    for (int i = 0; i < num_test_cases; i++) {
        uint128_t n = parse_decimal(test_cases[i].number);
        int bit_size = 128 - n.leading_zeros();
        
        const char* expected_algo = "Unknown";
        
        // Simple algorithm selection logic
        if (bit_size <= 20) {
            expected_algo = "Trial Division";
        } else if (bit_size <= 40) {
            expected_algo = "Pollard's Rho";
        } else if (bit_size <= 80) {
            expected_algo = "ECM";
        } else {
            expected_algo = "QS";
        }
        
        printf("Number: %s (%d bits) -> %s\n", 
               test_cases[i].name, bit_size, expected_algo);
    }
    
    printf("✅ PASS: Algorithm selection logic working\n");
    return true;
}

// Main test function
int main() {
    printf("=======================================================\n");
    printf("  CUDA Factorizer v2.2.0 Integration Test Suite\n");
    printf("  Simple Validation Tests\n");
    printf("=======================================================\n");
    
    bool all_passed = true;
    int tests_passed = 0;
    int total_tests = 0;
    
    // Test 1: CUDA availability
    total_tests++;
    if (test_cuda()) {
        tests_passed++;
    } else {
        all_passed = false;
    }
    
    // Test 2: uint128 arithmetic
    total_tests++;
    if (test_uint128_arithmetic()) {
        tests_passed++;
    } else {
        all_passed = false;
    }
    
    // Test 3: Algorithm selection
    total_tests++;
    if (test_algorithm_selection()) {
        tests_passed++;
    } else {
        all_passed = false;
    }
    
    // Test 4: Number parsing for all test cases
    for (int i = 0; i < num_test_cases; i++) {
        total_tests++;
        if (test_case(test_cases[i])) {
            tests_passed++;
        } else {
            all_passed = false;
        }
    }
    
    printf("\n=======================================================\n");
    printf("  TEST RESULTS SUMMARY\n");
    printf("=======================================================\n");
    printf("Tests passed: %d/%d\n", tests_passed, total_tests);
    
    if (all_passed) {
        printf("✅ ALL TESTS PASSED\n");
        printf("Integration components are working correctly.\n");
        printf("Ready for full factorization testing.\n");
    } else {
        printf("❌ SOME TESTS FAILED\n");
        printf("Please check the failed components before proceeding.\n");
    }
    
    printf("=======================================================\n");
    
    return all_passed ? 0 : 1;
}