/**
 * Montgomery Reduction for 128-bit Modular Arithmetic
 * High-performance modular arithmetic for repeated operations
 * Optimized for CUDA GTX 2070 architecture
 */

#ifndef MONTGOMERY_REDUCTION_CUH
#define MONTGOMERY_REDUCTION_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "uint128_improved.cuh"

// Montgomery context for 128-bit arithmetic
struct Montgomery128 {
    uint128_t n;          // modulus (must be odd)
    uint128_t r;          // R = 2^k where k is bit length of n
    uint128_t r_squared;   // R^2 mod n (for conversion to Montgomery form)
    uint128_t n_prime;    // -n^(-1) mod R
    int k;                // bit length of n
    
    __device__ __host__ void precompute() {
        // Ensure n is odd
        if ((n.low & 1) == 0) {
            printf("Error: Montgomery reduction requires odd modulus\n");
            return;
        }
        
        // Calculate bit length of n
        k = 128 - n.leading_zeros();
        
        // R = 2^k (we'll use k = 128 for simplicity)
        k = 128;
        r = uint128_t(0, 1ULL << 63);  // 2^127 for now
        
        // Calculate n' such that n * n' ≡ -1 (mod 2^64)
        // Using Hensel's lemma starting with 8-bit inverse
        uint64_t n_inv = n.low;
        
        // Newton-Raphson iteration for modular inverse
        // Start with a good initial guess
        uint64_t x = 1;
        for (int i = 0; i < 6; i++) {
            x = x * (2 - n_inv * x);
        }
        n_prime.low = -x;  // -n^(-1) mod 2^64
        n_prime.high = 0;
        
        // Calculate R^2 mod n for conversions
        // This is 2^256 mod n
        r_squared = calculate_r_squared();
    }
    
private:
    __device__ __host__ uint128_t calculate_r_squared() {
        // Calculate 2^256 mod n
        // Start with 2^128 and square it with reduction
        uint128_t result(0, 0);
        
        // First, calculate 2^128 mod n
        if (n.high == 0) {
            // Simple case for 64-bit modulus
            uint128_t two_128(0, 0x8000000000000000ULL);
            result = uint128_t(two_128.high % n.low, 0);
        } else {
            // General case: repeated doubling
            result = uint128_t(1, 0);
            for (int i = 0; i < 128; i++) {
                result = add_128(result, result);
                if (result >= n) {
                    result = subtract_128(result, n);
                }
            }
        }
        
        // Now square it to get 2^256 mod n
        uint256_t prod = multiply_128_128(result, result);
        
        // Reduce modulo n (simple method for setup)
        uint128_t r2 = uint128_t(prod.word[0], prod.word[1]);
        while (r2 >= n) {
            r2 = subtract_128(r2, n);
        }
        
        return r2;
    }
};

// Montgomery reduction: computes (T * R^(-1)) mod n
__device__ __host__ uint128_t montgomery_reduce(
    const uint256_t& T,
    const Montgomery128& mont
) {
    // Step 1: m = (T * n') mod R
    // Since R = 2^128, this is just the low 128 bits of T * n'
    uint128_t t_low(T.word[0], T.word[1]);
    uint256_t m_full = multiply_128_128(t_low, mont.n_prime);
    uint128_t m(m_full.word[0], m_full.word[1]);
    
    // Step 2: t = (T + m * n) / R
    // First compute m * n
    uint256_t mn = multiply_128_128(m, mont.n);
    
    // Add T + m * n
    uint64_t carry = 0;
    uint256_t sum;
    
    // Add word by word with carry
    for (int i = 0; i < 4; i++) {
        uint64_t temp = T.word[i] + mn.word[i] + carry;
        carry = (temp < T.word[i]) || (temp < mn.word[i]) ? 1 : 0;
        sum.word[i] = temp;
    }
    
    // Divide by R (shift right by 128 bits)
    uint128_t result(sum.word[2], sum.word[3]);
    
    // Step 3: Conditional subtraction
    if (result >= mont.n) {
        result = subtract_128(result, mont.n);
    }
    
    return result;
}

// Convert to Montgomery form: a * R mod n
__device__ __host__ uint128_t to_montgomery(
    const uint128_t& a,
    const Montgomery128& mont
) {
    // Compute a * R^2 * R^(-1) mod n = a * R mod n
    uint256_t prod = multiply_128_128(a, mont.r_squared);
    return montgomery_reduce(prod, mont);
}

// Convert from Montgomery form: a * R^(-1) mod n
__device__ __host__ uint128_t from_montgomery(
    const uint128_t& a_mont,
    const Montgomery128& mont
) {
    uint256_t T;
    T.word[0] = a_mont.low;
    T.word[1] = a_mont.high;
    T.word[2] = 0;
    T.word[3] = 0;
    return montgomery_reduce(T, mont);
}

// Montgomery multiplication: (a * b * R^(-1)) mod n
__device__ __host__ uint128_t montgomery_multiply(
    const uint128_t& a_mont,
    const uint128_t& b_mont,
    const Montgomery128& mont
) {
    uint256_t prod = multiply_128_128(a_mont, b_mont);
    return montgomery_reduce(prod, mont);
}

// Montgomery squaring (optimized)
__device__ __host__ uint128_t montgomery_square(
    const uint128_t& a_mont,
    const Montgomery128& mont
) {
    // Could be optimized further with specialized squaring
    return montgomery_multiply(a_mont, a_mont, mont);
}

// Montgomery exponentiation
__device__ uint128_t montgomery_exponent(
    const uint128_t& base,
    const uint128_t& exp,
    const Montgomery128& mont
) {
    // Convert base to Montgomery form
    uint128_t b_mont = to_montgomery(base, mont);
    
    // Result starts as 1 in Montgomery form (which is R mod n)
    uint128_t result_mont = to_montgomery(uint128_t(1, 0), mont);
    
    // Process exponent bits
    uint128_t e = exp;
    while (!e.is_zero()) {
        if (e.low & 1) {
            result_mont = montgomery_multiply(result_mont, b_mont, mont);
        }
        b_mont = montgomery_square(b_mont, mont);
        e = shift_right_128(e, 1);
    }
    
    // Convert back from Montgomery form
    return from_montgomery(result_mont, mont);
}

// Batch Montgomery operations kernel
__global__ void batch_montgomery_multiply(
    uint128_t* a_values,
    uint128_t* b_values,
    int count,
    uint128_t modulus,
    uint128_t* results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for Montgomery context
    __shared__ Montgomery128 shared_mont;
    
    // First thread initializes Montgomery parameters
    if (threadIdx.x == 0) {
        shared_mont.n = modulus;
        shared_mont.precompute();
    }
    __syncthreads();
    
    // Process values
    if (tid < count) {
        // Convert to Montgomery form
        uint128_t a_mont = to_montgomery(a_values[tid], shared_mont);
        uint128_t b_mont = to_montgomery(b_values[tid], shared_mont);
        
        // Multiply in Montgomery form
        uint128_t result_mont = montgomery_multiply(a_mont, b_mont, shared_mont);
        
        // Convert back
        results[tid] = from_montgomery(result_mont, shared_mont);
    }
}

// Performance comparison kernel
__global__ void benchmark_montgomery() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Test with a large prime modulus
    uint128_t n(0xFFFFFFFFFFFFFFC5ULL, 0xFFFFFFFFFFFFFFFFULL);  // Large prime
    
    // Setup Montgomery context
    Montgomery128 mont;
    mont.n = n;
    mont.precompute();
    
    // Test values
    uint128_t a(tid + 1, tid);
    uint128_t b(tid * 2 + 1, tid * 2);
    
    // Convert to Montgomery form
    uint128_t a_mont = to_montgomery(a, mont);
    uint128_t b_mont = to_montgomery(b, mont);
    
    // Benchmark Montgomery multiplication
    clock_t start = clock();
    
    uint128_t result = a_mont;
    #pragma unroll 4
    for (int i = 0; i < 1000; i++) {
        result = montgomery_multiply(result, b_mont, mont);
    }
    
    clock_t end = clock();
    
    // Convert back for verification
    result = from_montgomery(result, mont);
    
    // Report from first thread
    if (tid == 0) {
        double time_ms = (double)(end - start) / (double)CLOCKS_PER_SEC * 1000.0;
        printf("Montgomery benchmark: 1000 modmuls in %.3f ms\n", time_ms);
        printf("Result: %llx:%llx\n", result.high, result.low);
    }
}

// Test kernel for Montgomery reduction
__global__ void test_montgomery() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Testing Montgomery Reduction\n");
        
        // Test modulus (must be odd)
        uint128_t n(1000000007, 0);  // Small prime for easy verification
        
        Montgomery128 mont;
        mont.n = n;
        mont.precompute();
        
        // Test case 1: Simple multiplication
        uint128_t a(12345, 0);
        uint128_t b(67890, 0);
        
        // Expected: (12345 * 67890) mod 1000000007 = 838102050
        uint128_t a_mont = to_montgomery(a, mont);
        uint128_t b_mont = to_montgomery(b, mont);
        uint128_t result_mont = montgomery_multiply(a_mont, b_mont, mont);
        uint128_t result = from_montgomery(result_mont, mont);
        
        printf("Test 1: %llu * %llu mod %llu = %llu (expected: 838102050)\n",
               a.low, b.low, n.low, result.low);
        
        // Test case 2: Exponentiation
        uint128_t base(2, 0);
        uint128_t exp(100, 0);
        
        uint128_t exp_result = montgomery_exponent(base, exp, mont);
        printf("Test 2: 2^100 mod %llu = %llu\n", n.low, exp_result.low);
        
        // Test case 3: Identity test
        uint128_t one(1, 0);
        uint128_t one_mont = to_montgomery(one, mont);
        uint128_t one_back = from_montgomery(one_mont, mont);
        printf("Test 3: Identity test: 1 -> Montgomery -> back = %llu\n", one_back.low);
    }
}

#endif // MONTGOMERY_REDUCTION_CUH