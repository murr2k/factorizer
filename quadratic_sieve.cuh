/**
 * Quadratic Sieve Header File
 * GPU-accelerated factorization using the Quadratic Sieve algorithm
 */

#ifndef QUADRATIC_SIEVE_CUH
#define QUADRATIC_SIEVE_CUH

#include <cuda_runtime.h>
#include <vector>
#include "uint128_improved.cuh"

// Configuration parameters
#define QS_SIEVE_INTERVAL 1048576    // 2^20 - size of sieving interval
#define QS_MAX_FACTOR_BASE 4096      // Maximum primes in factor base
#define QS_MAX_RELATIONS 8192        // Maximum smooth relations to collect
#define QS_LOG_THRESHOLD 30          // Threshold for smooth number detection
#define QS_BLOCK_SIZE 256            // CUDA block size

// Structure for factor base prime
struct QSFactorBasePrime {
    uint32_t p;           // Prime number
    uint32_t root;        // Square root of n mod p
    float logp;           // log(p) for sieving
    uint32_t start_pos;   // Starting position in current interval
};

// Structure for smooth relation
struct QSRelation {
    uint128_t x;          // x value where Q(x) is smooth
    uint128_t Qx;         // Q(x) value
    uint32_t factors[64]; // Prime factors (indices into factor base)
    uint32_t num_factors; // Number of factors
};

// Structure for polynomial coefficients
struct QSPolynomial {
    int64_t a;            // Leading coefficient
    int64_t b;            // Linear coefficient
    uint128_t c;          // Constant term (related to n)
    std::vector<uint32_t> a_factors; // Prime factors of a
};

// Main factorization function
extern "C" bool quadratic_sieve_factor(uint128_t n, uint128_t& factor1, uint128_t& factor2);

// Helper functions
void qs_generate_factor_base(std::vector<QSFactorBasePrime>& factor_base, uint128_t n, uint32_t max_prime);
bool qs_is_smooth(uint128_t n, const std::vector<QSFactorBasePrime>& factor_base, std::vector<uint32_t>& factors);
void qs_generate_polynomial(QSPolynomial& poly, uint128_t n, const std::vector<QSFactorBasePrime>& factor_base);

// GPU kernels
__global__ void qs_sieve_kernel(
    QSFactorBasePrime* factor_base,
    uint32_t fb_size,
    float* sieve_array,
    uint32_t interval_size,
    int64_t interval_start,
    QSPolynomial poly
);

__global__ void qs_detect_smooth_kernel(
    float* sieve_array,
    uint32_t interval_size,
    int64_t interval_start,
    uint128_t n,
    QSPolynomial poly,
    QSRelation* relations,
    uint32_t* relation_count,
    uint32_t max_relations,
    float threshold
);

// Advanced features
class QuadraticSieve {
private:
    uint128_t n;
    std::vector<QSFactorBasePrime> factor_base;
    std::vector<QSRelation> relations;
    std::vector<QSPolynomial> polynomials;
    
    // Device pointers
    QSFactorBasePrime* d_factor_base;
    float* d_sieve_array;
    QSRelation* d_relations;
    uint32_t* d_relation_count;
    
    // Parameters
    uint32_t factor_base_size;
    uint32_t interval_size;
    float smooth_threshold;
    
public:
    QuadraticSieve(uint128_t number);
    ~QuadraticSieve();
    
    bool factor(uint128_t& factor1, uint128_t& factor2);
    void set_parameters(uint32_t fb_size, uint32_t interval, float threshold);
    
private:
    void initialize_factor_base();
    void sieve_interval(int64_t start);
    void collect_relations();
    bool solve_matrix(uint128_t& factor1, uint128_t& factor2);
    void switch_polynomial();
};

#endif // QUADRATIC_SIEVE_CUH