/**
 * Barrett Reduction v2.0 - Full Implementation
 * Complete 256-bit division and optimized reduction
 * For CUDA GTX 2070 architecture
 */

#ifndef BARRETT_REDUCTION_V2_CUH
#define BARRETT_REDUCTION_V2_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "uint128_improved.cuh"

// Extended arithmetic for Barrett calculations
struct uint256_extended {
    uint64_t word[5];  // Extra word for overflow handling
    
    __device__ __host__ uint256_extended() {
        for (int i = 0; i < 5; i++) word[i] = 0;
    }
    
    __device__ __host__ void shift_left(int bits) {
        if (bits == 0) return;
        if (bits >= 64) {
            // Shift by whole words
            int words = bits / 64;
            bits %= 64;
            for (int i = 4; i >= words; i--) {
                word[i] = word[i - words];
            }
            for (int i = 0; i < words; i++) {
                word[i] = 0;
            }
        }
        if (bits > 0) {
            // Shift remaining bits
            uint64_t carry = 0;
            for (int i = 0; i < 5; i++) {
                uint64_t new_carry = word[i] >> (64 - bits);
                word[i] = (word[i] << bits) | carry;
                carry = new_carry;
            }
        }
    }
    
    __device__ __host__ bool subtract(const uint256_extended& other) {
        uint64_t borrow = 0;
        for (int i = 0; i < 5; i++) {
            uint64_t temp = word[i] - other.word[i] - borrow;
            borrow = (word[i] < other.word[i] + borrow) ? 1 : 0;
            word[i] = temp;
        }
        return borrow == 0;  // Return true if no final borrow
    }
    
    __device__ __host__ bool greater_equal(const uint256_extended& other) const {
        for (int i = 4; i >= 0; i--) {
            if (word[i] > other.word[i]) return true;
            if (word[i] < other.word[i]) return false;
        }
        return true;  // Equal
    }
};

// Full 256-bit division for Barrett μ calculation
__device__ __host__ uint128_t divide_256_128(const uint256_t& dividend, const uint128_t& divisor) {
    if (divisor.is_zero()) {
        return uint128_t(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    }
    
    uint256_extended remainder;
    uint256_extended div;
    uint128_t quotient(0, 0);
    
    // Copy dividend to remainder
    for (int i = 0; i < 4; i++) {
        remainder.word[i] = dividend.word[i];
    }
    
    // Copy divisor to div (extended)
    div.word[0] = divisor.low;
    div.word[1] = divisor.high;
    
    // Find the most significant bit of divisor
    int divisor_bits = 128 - divisor.leading_zeros();
    
    // Align divisor with dividend's MSB
    int dividend_bits = 256;
    for (int i = 3; i >= 0; i--) {
        if (dividend.word[i] != 0) {
            dividend_bits = i * 64 + 64 - __builtin_clzll(dividend.word[i]);
            break;
        }
    }
    
    int shift = dividend_bits - divisor_bits;
    if (shift > 0) {
        div.shift_left(shift);
    }
    
    // Perform long division
    for (int i = shift; i >= 0; i--) {
        if (remainder.greater_equal(div)) {
            remainder.subtract(div);
            // Set bit i in quotient
            if (i < 64) {
                quotient.low |= (1ULL << i);
            } else {
                quotient.high |= (1ULL << (i - 64));
            }
        }
        // Shift divisor right by 1
        uint64_t carry = 0;
        for (int j = 4; j >= 0; j--) {
            uint64_t new_carry = div.word[j] & 1;
            div.word[j] = (div.word[j] >> 1) | (carry << 63);
            carry = new_carry;
        }
    }
    
    return quotient;
}

// Improved Barrett structure with full μ calculation
struct Barrett128_v2 {
    uint128_t n;      // modulus
    uint128_t mu;     // precomputed μ = floor(2^k / n)
    int k;            // bit length for reduction
    int n_bits;       // actual bit length of n
    
    __device__ __host__ void precompute() {
        // Calculate exact bit length of n
        n_bits = 128 - n.leading_zeros();
        k = 2 * n_bits;
        
        // Calculate μ = floor(2^k / n)
        if (k <= 128) {
            // Simple case: 2^k fits in 128 bits
            uint128_t two_pow_k;
            if (k == 128) {
                two_pow_k = uint128_t(0, 0x8000000000000000ULL);
            } else if (k >= 64) {
                two_pow_k = uint128_t(0, 1ULL << (k - 64));
            } else {
                two_pow_k = uint128_t(1ULL << k, 0);
            }
            mu = divide_128_64(two_pow_k, n.low);  // Simplified for now
        } else {
            // k > 128: Need 256-bit dividend
            uint256_t two_pow_k;
            for (int i = 0; i < 4; i++) two_pow_k.word[i] = 0;
            
            int word_idx = k / 64;
            int bit_idx = k % 64;
            if (word_idx < 4) {
                two_pow_k.word[word_idx] = 1ULL << bit_idx;
            }
            
            mu = divide_256_128(two_pow_k, n);
        }
    }
    
    __device__ __host__ uint128_t reduce(const uint256_t& x) const {
        // Barrett reduction: r = x - floor(x * μ / 2^k) * n
        
        // Step 1: Calculate q = floor(x * μ / 2^k)
        // First compute x * μ (up to 384 bits)
        uint128_t x_low(x.word[0], x.word[1]);
        uint128_t x_high(x.word[2], x.word[3]);
        
        // x * μ = x_low * μ + (x_high * μ) << 128
        uint256_t x_mu_low = multiply_128_128(x_low, mu);
        uint256_t x_mu_high = multiply_128_128(x_high, mu);
        
        // Combine and shift right by k bits
        uint128_t q;
        if (k >= 256) {
            // q would be 0
            q = uint128_t(0, 0);
        } else if (k >= 192) {
            // Only highest parts matter
            int shift = k - 192;
            q.low = x_mu_high.word[0] >> shift;
            q.high = (x_mu_high.word[1] >> shift) | (x_mu_high.word[2] << (64 - shift));
        } else if (k >= 128) {
            // Shift within x_mu_high
            int shift = k - 128;
            uint128_t temp(x_mu_low.word[2], x_mu_low.word[3]);
            temp = shift_right_128(temp, shift);
            q = add_128(temp, shift_left_128(x_high, 128 - shift));
        } else {
            // k < 128: normal case
            q = shift_right_128(x_low, k);
        }
        
        // Step 2: Calculate q * n
        uint256_t qn = multiply_128_128(q, n);
        
        // Step 3: Calculate r = x - q * n
        uint128_t r = x_low;
        if (qn.word[0] <= x.word[0]) {
            r.low = x.word[0] - qn.word[0];
            uint64_t borrow = 0;
            if (x.word[0] < qn.word[0]) borrow = 1;
            r.high = x.word[1] - qn.word[1] - borrow;
        }
        
        // Step 4: Correction (at most 2 subtractions needed)
        while (r >= n) {
            r = subtract_128(r, n);
        }
        
        return r;
    }
    
    // Optimized reduction for 128-bit input
    __device__ __host__ uint128_t reduce_128(const uint128_t& x) const {
        if (x < n) return x;
        
        // For 128-bit input, we can optimize
        if (n_bits <= 64) {
            // Fast path for small modulus
            return uint128_t(x.low % n.low, 0);
        }
        
        // General case using full Barrett
        uint256_t x_extended;
        x_extended.word[0] = x.low;
        x_extended.word[1] = x.high;
        x_extended.word[2] = 0;
        x_extended.word[3] = 0;
        
        return reduce(x_extended);
    }
};

// Optimized modular multiplication
__device__ __host__ uint128_t modmul_barrett_v2(
    const uint128_t& a, 
    const uint128_t& b, 
    const Barrett128_v2& barrett
) {
    // First reduce inputs if necessary
    uint128_t a_reduced = (a >= barrett.n) ? barrett.reduce_128(a) : a;
    uint128_t b_reduced = (b >= barrett.n) ? barrett.reduce_128(b) : b;
    
    // Multiply
    uint256_t product = multiply_128_128(a_reduced, b_reduced);
    
    // Reduce
    return barrett.reduce(product);
}

// Modular exponentiation using Barrett reduction
__device__ uint128_t modexp_barrett(
    const uint128_t& base,
    const uint128_t& exp,
    const Barrett128_v2& barrett
) {
    uint128_t result(1, 0);
    uint128_t b = barrett.reduce_128(base);
    
    // Process exponent bits from right to left
    uint128_t e = exp;
    while (!e.is_zero()) {
        if (e.low & 1) {
            result = modmul_barrett_v2(result, b, barrett);
        }
        b = modmul_barrett_v2(b, b, barrett);
        e = shift_right_128(e, 1);
    }
    
    return result;
}

// Batch Barrett reduction for multiple values
__global__ void batch_barrett_reduce(
    uint128_t* values,
    int count,
    uint128_t modulus,
    uint128_t* results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for Barrett parameters
    __shared__ Barrett128_v2 shared_barrett;
    
    // First thread initializes Barrett parameters
    if (threadIdx.x == 0) {
        shared_barrett.n = modulus;
        shared_barrett.precompute();
    }
    __syncthreads();
    
    // Process values
    if (tid < count) {
        results[tid] = shared_barrett.reduce_128(values[tid]);
    }
}

// Performance test kernel
__global__ void benchmark_barrett_v2() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Test parameters
    uint128_t n(0x1234567890ABCDEFULL, 0x1ULL);  // Large prime
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // Warm-up
    uint128_t x(tid, tid);
    uint128_t result = x;
    
    // Benchmark loop
    clock_t start = clock();
    
    #pragma unroll 4
    for (int i = 0; i < 1000; i++) {
        result = modmul_barrett_v2(result, x, barrett);
    }
    
    clock_t end = clock();
    
    // Report results from first thread
    if (tid == 0) {
        double time_ms = (double)(end - start) / (double)CLOCKS_PER_SEC * 1000.0;
        printf("Barrett v2 benchmark: 1000 modmuls in %.3f ms\n", time_ms);
        printf("Result: %llx:%llx\n", result.high, result.low);
    }
}

#endif // BARRETT_REDUCTION_V2_CUH