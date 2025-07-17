/**
 * Test program for complete Quadratic Sieve implementation
 */

#include <cstdio>
#include <cstring>
#include <chrono>
#include "quadratic_sieve_complete.cuh"

// Function to print uint128 in decimal
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

// Test individual components
void test_polynomial_generation() {
    printf("\n=== Testing Polynomial Generation ===\n");
    
    uint128_t n = parse_decimal("1208925819614629174706449");
    QSContext* ctx = qs_create_context(n);
    
    // Generate factor base
    qs_generate_factor_base(ctx);
    printf("Factor base size: %u\n", ctx->factor_base_size);
    
    // Generate multiple polynomials
    for (int i = 0; i < 5; i++) {
        if (qs_generate_polynomial(ctx, i)) {
            printf("\nPolynomial %d:\n", i);
            printf("  a = ");
            print_uint128_decimal(ctx->polynomials[i].a);
            printf("\n  b = ");
            print_uint128_decimal(ctx->polynomials[i].b);
            printf("\n  c = ");
            print_uint128_decimal(ctx->polynomials[i].c);
            printf("\n  factors of a: %u\n", ctx->polynomials[i].num_factors);
        }
    }
    
    qs_destroy_context(ctx);
}

// Test matrix solving
void test_matrix_solving() {
    printf("\n=== Testing Matrix Solving ===\n");
    
    // Create a small test matrix over GF(2)
    QSContext* ctx = new QSContext;
    ctx->matrix.num_rows = 5;
    ctx->matrix.num_cols = 4;
    ctx->matrix.rows = new uint32_t*[5];
    ctx->matrix.row_sizes = new uint32_t[5];
    
    // Example matrix (each row is list of column indices with 1s)
    uint32_t row0[] = {0, 1, 3};
    uint32_t row1[] = {1, 2};
    uint32_t row2[] = {0, 2, 3};
    uint32_t row3[] = {0, 1, 2};
    uint32_t row4[] = {1, 3};
    
    ctx->matrix.rows[0] = row0; ctx->matrix.row_sizes[0] = 3;
    ctx->matrix.rows[1] = row1; ctx->matrix.row_sizes[1] = 2;
    ctx->matrix.rows[2] = row2; ctx->matrix.row_sizes[2] = 3;
    ctx->matrix.rows[3] = row3; ctx->matrix.row_sizes[3] = 3;
    ctx->matrix.rows[4] = row4; ctx->matrix.row_sizes[4] = 2;
    
    // Add some dummy relations
    ctx->smooth_relations.resize(5);
    
    std::vector<std::vector<int>> dependencies;
    if (qs_solve_matrix(ctx, dependencies)) {
        printf("Found %zu dependencies\n", dependencies.size());
        for (size_t i = 0; i < dependencies.size(); i++) {
            printf("Dependency %zu: ", i);
            for (int idx : dependencies[i]) {
                printf("%d ", idx);
            }
            printf("\n");
        }
    }
    
    delete ctx;
}

// Main test function
int main(int argc, char* argv[]) {
    printf("Quadratic Sieve Complete - Test Program\n");
    printf("=======================================\n");
    
    // Initialize CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    
    if (argc > 1 && strcmp(argv[1], "components") == 0) {
        // Test individual components
        test_polynomial_generation();
        test_matrix_solving();
    } else if (argc > 1) {
        // Factor specific number
        uint128_t n = parse_decimal(argv[1]);
        printf("\nFactoring: ");
        print_uint128_decimal(n);
        printf("\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        uint128_t factor1, factor2;
        bool success = quadratic_sieve_factor_complete(n, factor1, factor2);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        if (success) {
            printf("\nFactorization successful!\n");
            printf("Factor 1: ");
            print_uint128_decimal(factor1);
            printf("\nFactor 2: ");
            print_uint128_decimal(factor2);
            printf("\n");
            
            // Verify
            uint256_t product = multiply_128_128(factor1, factor2);
            uint128_t n_check(product.word[0], product.word[1]);
            
            if (n_check == n) {
                printf("Verification: PASSED\n");
            } else {
                printf("Verification: FAILED\n");
            }
        } else {
            printf("\nFactorization failed\n");
        }
        
        printf("Time: %.3f seconds\n", elapsed);
    } else {
        // Default test cases
        struct TestCase {
            const char* n_str;
            const char* description;
        };
        
        TestCase tests[] = {
            {"143", "Small test (11 × 13)"},
            {"8633", "Medium test (89 × 97)"},
            {"1073676289", "32-bit test (32713 × 32833)"},
            {"1152921504606846999", "60-bit test"},
            {"29318992932113061061655073", "86-bit test"}
        };
        
        for (const auto& test : tests) {
            printf("\n%s\n", test.description);
            printf("N = %s\n", test.n_str);
            
            uint128_t n = parse_decimal(test.n_str);
            uint128_t factor1, factor2;
            
            auto start = std::chrono::high_resolution_clock::now();
            bool success = quadratic_sieve_factor_complete(n, factor1, factor2);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (success) {
                printf("Factors: ");
                print_uint128_decimal(factor1);
                printf(" × ");
                print_uint128_decimal(factor2);
                printf("\n");
            } else {
                printf("Failed to factor\n");
            }
            
            printf("Time: %.3f seconds\n", 
                   std::chrono::duration<double>(end - start).count());
        }
    }
    
    return 0;
}