/**
 * Quadratic Sieve Complete Implementation Header
 * GPU-accelerated factorization using the Quadratic Sieve algorithm
 */

#ifndef QUADRATIC_SIEVE_COMPLETE_CUH
#define QUADRATIC_SIEVE_COMPLETE_CUH

#include <cuda_runtime.h>
#include <vector>
#include "uint128_improved.cuh"

// Configuration parameters
#define QS_SIEVE_INTERVAL 1048576    // 2^20 - size of sieving interval
#define QS_MAX_FACTOR_BASE 8192      // Maximum primes in factor base
#define QS_MAX_RELATIONS 16384       // Maximum smooth relations to collect
#define QS_LOG_THRESHOLD 28          // Threshold for smooth number detection
#define QS_BLOCK_SIZE 256            // CUDA block size
#define QS_MAX_POLY_FACTORS 32       // Maximum factors for polynomial generation

// Structure for factor base prime
struct QSFactorBasePrime {
    uint32_t p;           // Prime number
    uint32_t root;        // Square root of n mod p
    float logp;           // log(p) for sieving
    uint32_t ainv_root;   // Root adjusted for polynomial a
};

// Structure for smooth relation
struct QSRelation {
    uint128_t x;          // x value where Q(x) is smooth
    uint128_t Qx;         // Q(x) value
    uint32_t* exponents;  // Exponent vector for matrix
    uint32_t large_prime; // Large prime if partial relation
    bool is_partial;      // Flag for partial relations
};

// Structure for polynomial coefficients
struct QSPolynomial {
    uint128_t a;          // Leading coefficient
    uint128_t b;          // Linear coefficient
    uint128_t c;          // Constant term (related to n)
    uint32_t* a_factors;  // Prime factors of a
    uint32_t num_factors; // Number of factors in a
    uint128_t sqrt_a;     // Square root of a
};

// Device structure for sieving
struct QSSieveData {
    QSFactorBasePrime* factor_base;
    uint32_t fb_size;
    uint128_t n;
    QSPolynomial* poly;
    float* sieve_array;
    int64_t interval_start;
    uint32_t interval_size;
    uint32_t* smooth_indices;
    uint32_t* smooth_count;
};

// Matrix structure for linear algebra
struct QSMatrix {
    uint32_t** rows;      // Sparse matrix representation
    uint32_t num_rows;    // Number of relations
    uint32_t num_cols;    // Number of primes in factor base
    uint32_t* row_sizes;  // Number of non-zero entries per row
};

// Main QS context
struct QSContext {
    uint128_t n;
    uint32_t factor_base_size;
    uint32_t target_relations;
    std::vector<QSFactorBasePrime> factor_base;
    std::vector<QSRelation> smooth_relations;
    std::vector<QSRelation> partial_relations;
    std::vector<QSPolynomial> polynomials;
    QSMatrix matrix;
    
    // GPU resources
    QSFactorBasePrime* d_factor_base;
    float* d_sieve_array;
    uint32_t* d_smooth_indices;
    uint32_t* d_smooth_count;
    QSPolynomial* d_polynomial;
    
    // Statistics
    uint64_t total_sieved;
    uint32_t polynomial_count;
    double sieving_time;
    double matrix_time;
};

// Function declarations
extern "C" {
    // Main entry point
    bool quadratic_sieve_factor_complete(uint128_t n, uint128_t& factor1, uint128_t& factor2);
    
    // Context management
    QSContext* qs_create_context(uint128_t n);
    void qs_destroy_context(QSContext* ctx);
    
    // Core algorithms
    bool qs_generate_factor_base(QSContext* ctx);
    bool qs_generate_polynomial(QSContext* ctx, int poly_index);
    bool qs_sieve_interval(QSContext* ctx, int64_t start, uint32_t size);
    bool qs_build_matrix(QSContext* ctx);
    bool qs_solve_matrix(QSContext* ctx, std::vector<std::vector<int>>& dependencies);
    bool qs_extract_factors(QSContext* ctx, const std::vector<std::vector<int>>& deps, 
                           uint128_t& factor1, uint128_t& factor2);
    
    // GPU kernels
    __global__ void qs_sieve_kernel_optimized(QSSieveData data);
    __global__ void qs_detect_smooth_kernel_optimized(QSSieveData data);
    __global__ void qs_trial_divide_kernel(uint128_t* values, uint32_t count,
                                          QSFactorBasePrime* fb, uint32_t fb_size,
                                          QSRelation* relations, uint32_t* rel_count);
}

// Inline helper functions
__device__ __host__ inline uint32_t qs_mod_inverse(uint32_t a, uint32_t m) {
    int32_t t = 0, newt = 1;
    int32_t r = m, newr = a;
    
    while (newr != 0) {
        int32_t quotient = r / newr;
        int32_t temp = t;
        t = newt;
        newt = temp - quotient * newt;
        temp = r;
        r = newr;
        newr = temp - quotient * newr;
    }
    
    if (r > 1) return 0; // a is not invertible
    if (t < 0) t += m;
    
    return (uint32_t)t;
}

#endif // QUADRATIC_SIEVE_COMPLETE_CUH