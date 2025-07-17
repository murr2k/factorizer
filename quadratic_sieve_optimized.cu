/**
 * Optimized Quadratic Sieve Implementation
 * Features:
 * - Multiple polynomial support (MPQS)
 * - Self-initializing polynomials
 * - Optimized GPU memory access patterns
 * - Batch processing of relations
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include "uint128_improved.cuh"
#include "quadratic_sieve.cuh"

// Optimized configuration
#define QS_LARGE_PRIME_BOUND 1000000
#define QS_POLY_A_BITS 32
#define QS_BATCH_SIZE 32
#define QS_SHARED_MEM_SIZE 4096

/**
 * Self-initializing polynomial generation
 * Generates polynomials of form Q(x) = (Ax + B)^2 - N
 */
class PolynomialGenerator {
private:
    uint128_t n;
    uint64_t sqrt_n;
    std::vector<uint32_t> prime_list;
    std::vector<uint32_t> current_factors;
    uint64_t target_a_size;
    
public:
    PolynomialGenerator(uint128_t number) : n(number) {
        sqrt_n = isqrt(n.low);
        target_a_size = 1ULL << (QS_POLY_A_BITS / 2);
    }
    
    __host__ void initialize(const std::vector<QSFactorBasePrime>& factor_base) {
        // Select primes for polynomial 'a' coefficients
        prime_list.clear();
        for (const auto& fbp : factor_base) {
            if (fbp.p > 100 && fbp.p < 10000) {
                prime_list.push_back(fbp.p);
            }
        }
    }
    
    __host__ bool generate_next(QSPolynomial& poly) {
        // Generate 'a' as product of primes close to target size
        poly.a = 1;
        poly.a_factors.clear();
        
        // Randomly select primes
        for (int i = 0; i < 3 && poly.a < target_a_size; i++) {
            uint32_t idx = rand() % prime_list.size();
            uint32_t p = prime_list[idx];
            poly.a *= p;
            poly.a_factors.push_back(p);
        }
        
        // Compute B such that B^2 ≡ N (mod A)
        // Using Hensel lifting for composite modulus
        poly.b = compute_b_coefficient(poly.a);
        
        // Set c = n for the polynomial form
        poly.c = n;
        
        return true;
    }
    
private:
    __host__ int64_t compute_b_coefficient(int64_t a) {
        // Simplified: should use Hensel lifting
        // For now, use approximation
        int64_t b = sqrt_n % a;
        
        // Adjust b to minimize |b|
        if (b > a / 2) b -= a;
        
        return b;
    }
    
    __host__ uint64_t isqrt(uint64_t n) {
        if (n == 0) return 0;
        uint64_t x = n;
        uint64_t y = (x + 1) / 2;
        while (y < x) {
            x = y;
            y = (x + n / x) / 2;
        }
        return x;
    }
};

/**
 * Optimized GPU sieving kernel with shared memory
 */
__global__ void qs_sieve_optimized_kernel(
    QSFactorBasePrime* factor_base,
    uint32_t fb_size,
    float* sieve_array,
    uint32_t interval_size,
    int64_t interval_start,
    int64_t poly_a,
    int64_t poly_b
) {
    // Shared memory for caching factor base
    extern __shared__ QSFactorBasePrime shared_fb[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperatively load factor base into shared memory
    for (int i = tid; i < fb_size && i < QS_SHARED_MEM_SIZE / sizeof(QSFactorBasePrime); i += blockDim.x) {
        shared_fb[i] = factor_base[i];
    }
    __syncthreads();
    
    // Each thread processes multiple positions
    int positions_per_thread = (interval_size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int start_pos = gid * positions_per_thread;
    int end_pos = min(start_pos + positions_per_thread, (int)interval_size);
    
    // Use registers for accumulation
    for (int pos = start_pos; pos < end_pos; pos++) {
        float log_sum = 0.0f;
        
        // Check divisibility by small primes
        int64_t x = interval_start + pos;
        int64_t qx_linear = poly_a * x + poly_b;
        
        // Process primes from shared memory
        int shared_fb_size = min((int)fb_size, (int)(QS_SHARED_MEM_SIZE / sizeof(QSFactorBasePrime)));
        for (int i = 0; i < shared_fb_size; i++) {
            uint32_t p = shared_fb[i].p;
            
            // Check if Q(x) ≡ 0 (mod p)
            if ((qx_linear % p) == shared_fb[i].root || 
                (qx_linear % p) == (p - shared_fb[i].root)) {
                log_sum += shared_fb[i].logp;
            }
        }
        
        // Process remaining primes from global memory
        for (int i = shared_fb_size; i < fb_size; i++) {
            uint32_t p = factor_base[i].p;
            if ((qx_linear % p) == factor_base[i].root || 
                (qx_linear % p) == (p - factor_base[i].root)) {
                log_sum += factor_base[i].logp;
            }
        }
        
        // Write result
        sieve_array[pos] = log_sum;
    }
}

/**
 * Batch smooth detection kernel
 */
__global__ void qs_batch_smooth_detect_kernel(
    float* sieve_array,
    uint32_t interval_size,
    int64_t interval_start,
    uint128_t n,
    int64_t poly_a,
    int64_t poly_b,
    QSRelation* relations,
    uint32_t* relation_count,
    uint32_t max_relations,
    float threshold,
    uint32_t* smooth_positions,
    uint32_t* smooth_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // First pass: identify smooth candidates
    for (uint32_t i = tid; i < interval_size; i += stride) {
        if (sieve_array[i] >= threshold) {
            uint32_t idx = atomicAdd(smooth_count, 1);
            if (idx < max_relations) {
                smooth_positions[idx] = i;
            }
        }
    }
    __syncthreads();
    
    // Second pass: compute Q(x) values for smooth candidates
    uint32_t total_smooth = *smooth_count;
    for (uint32_t i = tid; i < total_smooth && i < max_relations; i += stride) {
        uint32_t pos = smooth_positions[i];
        int64_t x = interval_start + pos;
        
        // Compute Q(x) = (ax + b)^2 - n
        uint128_t ax_b;
        ax_b.low = poly_a * x + poly_b;
        ax_b.high = 0;
        
        // Square it
        uint128_t qx = ax_b * ax_b;
        
        // Subtract n
        if (qx >= n) {
            qx = qx - n;
            
            // Store relation
            uint32_t idx = atomicAdd(relation_count, 1);
            if (idx < max_relations) {
                relations[idx].x = uint128_t(x);
                relations[idx].Qx = qx;
                relations[idx].num_factors = 0;
            }
        }
    }
}

/**
 * Main optimized Quadratic Sieve class implementation
 */
QuadraticSieve::QuadraticSieve(uint128_t number) : n(number) {
    // Set default parameters based on size of n
    double log_n = log2(n.low);
    factor_base_size = (uint32_t)(exp(sqrt(log_n * log(log_n))) * 1.2);
    factor_base_size = min(factor_base_size, QS_MAX_FACTOR_BASE);
    
    interval_size = QS_SIEVE_INTERVAL;
    smooth_threshold = QS_LOG_THRESHOLD;
    
    // Allocate device memory
    cudaMalloc(&d_factor_base, factor_base_size * sizeof(QSFactorBasePrime));
    cudaMalloc(&d_sieve_array, interval_size * sizeof(float));
    cudaMalloc(&d_relations, QS_MAX_RELATIONS * sizeof(QSRelation));
    cudaMalloc(&d_relation_count, sizeof(uint32_t));
}

QuadraticSieve::~QuadraticSieve() {
    cudaFree(d_factor_base);
    cudaFree(d_sieve_array);
    cudaFree(d_relations);
    cudaFree(d_relation_count);
}

void QuadraticSieve::initialize_factor_base() {
    // Generate factor base
    qs_generate_factor_base(factor_base, n, factor_base_size * 10);
    
    // Truncate to desired size
    if (factor_base.size() > factor_base_size) {
        factor_base.resize(factor_base_size);
    }
    
    // Copy to device
    cudaMemcpy(d_factor_base, factor_base.data(),
               factor_base.size() * sizeof(QSFactorBasePrime), cudaMemcpyHostToDevice);
}

void QuadraticSieve::sieve_interval(int64_t start) {
    // Clear sieve array
    cudaMemset(d_sieve_array, 0, interval_size * sizeof(float));
    
    // Launch optimized sieving kernel
    int blocks = 256;
    int threads = QS_BLOCK_SIZE;
    size_t shared_mem = QS_SHARED_MEM_SIZE;
    
    // Use current polynomial
    QSPolynomial& poly = polynomials.back();
    
    qs_sieve_optimized_kernel<<<blocks, threads, shared_mem>>>(
        d_factor_base, factor_base.size(), d_sieve_array, interval_size,
        start, poly.a, poly.b
    );
    cudaDeviceSynchronize();
}

bool QuadraticSieve::factor(uint128_t& factor1, uint128_t& factor2) {
    printf("Optimized Quadratic Sieve starting...\n");
    printf("n = %llu (high: %llu)\n", n.low, n.high);
    
    // Initialize
    initialize_factor_base();
    printf("Factor base: %zu primes\n", factor_base.size());
    
    // Polynomial generator
    PolynomialGenerator poly_gen(n);
    poly_gen.initialize(factor_base);
    
    // Main sieving loop
    uint64_t sqrt_n = isqrt(n.low);
    int64_t sieve_radius = interval_size * 10;
    
    uint32_t* d_smooth_positions;
    uint32_t* d_smooth_count;
    cudaMalloc(&d_smooth_positions, QS_MAX_RELATIONS * sizeof(uint32_t));
    cudaMalloc(&d_smooth_count, sizeof(uint32_t));
    
    while (relations.size() < factor_base.size() + 100) {
        // Generate new polynomial
        QSPolynomial poly;
        if (!poly_gen.generate_next(poly)) {
            break;
        }
        polynomials.push_back(poly);
        
        printf("Polynomial %zu: a=%lld, b=%lld\n", polynomials.size(), poly.a, poly.b);
        
        // Sieve multiple intervals
        for (int side = -1; side <= 1; side += 2) {
            for (int64_t offset = 0; offset < sieve_radius; offset += interval_size) {
                int64_t start = sqrt_n + side * offset;
                
                // Sieve interval
                sieve_interval(start);
                
                // Detect smooth numbers
                cudaMemset(d_relation_count, 0, sizeof(uint32_t));
                cudaMemset(d_smooth_count, 0, sizeof(uint32_t));
                
                int blocks = 256;
                qs_batch_smooth_detect_kernel<<<blocks, QS_BLOCK_SIZE>>>(
                    d_sieve_array, interval_size, start, n, poly.a, poly.b,
                    d_relations, d_relation_count, QS_MAX_RELATIONS, smooth_threshold,
                    d_smooth_positions, d_smooth_count
                );
                cudaDeviceSynchronize();
                
                // Collect relations
                collect_relations();
                
                if (relations.size() >= factor_base.size() + 100) {
                    break;
                }
            }
        }
    }
    
    printf("Collected %zu relations\n", relations.size());
    
    // Cleanup temporary allocations
    cudaFree(d_smooth_positions);
    cudaFree(d_smooth_count);
    
    // Matrix solving
    bool success = solve_matrix(factor1, factor2);
    
    return success;
}

void QuadraticSieve::collect_relations() {
    uint32_t count;
    cudaMemcpy(&count, d_relation_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    if (count > 0) {
        std::vector<QSRelation> new_relations(count);
        cudaMemcpy(new_relations.data(), d_relations,
                   count * sizeof(QSRelation), cudaMemcpyDeviceToHost);
        
        // Verify smoothness and factor
        for (auto& rel : new_relations) {
            std::vector<uint32_t> factors;
            if (qs_is_smooth(rel.Qx, factor_base, factors)) {
                rel.num_factors = factors.size();
                for (size_t i = 0; i < factors.size() && i < 64; i++) {
                    rel.factors[i] = factors[i];
                }
                relations.push_back(rel);
            }
        }
    }
}

bool QuadraticSieve::solve_matrix(uint128_t& factor1, uint128_t& factor2) {
    printf("Matrix solving for %zu relations...\n", relations.size());
    
    // TODO: Implement Gaussian elimination over GF(2)
    // This would create a matrix where each row represents a relation
    // and each column represents a prime in the factor base
    // The goal is to find a subset of relations whose product is a square
    
    printf("Matrix solving not yet implemented\n");
    return false;
}

// Helper function implementations
void qs_generate_factor_base(std::vector<QSFactorBasePrime>& factor_base, uint128_t n, uint32_t max_prime) {
    // Implementation from quadratic_sieve_core.cu
    std::vector<bool> is_prime(max_prime + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    for (uint32_t i = 2; i * i <= max_prime; i++) {
        if (is_prime[i]) {
            for (uint32_t j = i * i; j <= max_prime; j += i) {
                is_prime[j] = false;
            }
        }
    }
    
    uint64_t n_low = n.low;
    factor_base.clear();
    factor_base.push_back({2, 1, log2f(2.0f), 0});
    
    for (uint32_t p = 3; p <= max_prime && factor_base.size() < QS_MAX_FACTOR_BASE; p += 2) {
        if (is_prime[p]) {
            if (mod_pow((uint32_t)(n_low % p), (p - 1) / 2, p) == 1) {
                uint32_t root = tonelli_shanks(n_low, p);
                if (root > 0) {
                    factor_base.push_back({p, root, log2f((float)p), 0});
                }
            }
        }
    }
}

bool qs_is_smooth(uint128_t n, const std::vector<QSFactorBasePrime>& factor_base, std::vector<uint32_t>& factors) {
    factors.clear();
    uint64_t remaining = n.low;
    
    for (size_t i = 0; i < factor_base.size() && remaining > 1; i++) {
        uint32_t p = factor_base[i].p;
        while (remaining % p == 0) {
            remaining /= p;
            factors.push_back(i);
        }
    }
    
    // Check for single large prime
    if (remaining > 1 && remaining < QS_LARGE_PRIME_BOUND) {
        // Large prime variation - store for later combining
        return true;
    }
    
    return remaining == 1;
}