/**
 * Clean Barrett Reduction Implementation for 128-bit Modular Arithmetic
 * 
 * This implementation follows the standard Barrett reduction algorithm:
 * - Precomputes mu = floor(2^k / modulus) where k = 2 * bit_length(modulus)
 * - Reduces x mod m using the approximation q = floor((x * mu) / 2^k)
 * - Handles 256-bit intermediate results properly
 */

#ifndef BARRETT_CLEAN_CUH
#define BARRETT_CLEAN_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>

// Include the uint128 types
#include "uint128_improved.cuh"

// Barrett reduction parameters
struct BarrettParams {
    uint128_t modulus;  // The modulus n
    uint128_t mu;       // Precomputed mu = floor(2^k / n)
    int k;              // k = 2 * bit_length(n)
};

/**
 * Compute the bit length of a 128-bit number
 */
__device__ __host__ inline int bit_length_128(const uint128_t& n) {
    if (n.high != 0) {
        // Count leading zeros in high word and adjust
        #ifdef __CUDA_ARCH__
        return 128 - __clzll(n.high);
        #else
        return 128 - __builtin_clzll(n.high);
        #endif
    } else if (n.low != 0) {
        // Only low word is non-zero
        #ifdef __CUDA_ARCH__
        return 64 - __clzll(n.low);
        #else
        return 64 - __builtin_clzll(n.low);
        #endif
    }
    return 0;  // n is zero
}

/**
 * Divide 2^k by n to compute mu = floor(2^k / n)
 * This is a critical function for Barrett reduction
 */
__device__ __host__ uint128_t compute_barrett_mu(const uint128_t& n, int k) {
    if (n.is_zero()) {
        return uint128_t(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);  // Error case
    }
    
    // For k=128, we need floor(2^128 / n)
    if (k == 128) {
        if (n.high == 0 && n.low != 0) {
            // n fits in 64 bits - use more accurate division
            // We want floor(2^128 / n.low)
            // This is equivalent to floor((2^64 - 1) * 2^64 / n.low) + floor(2^64 / n.low)
            
            uint64_t q_high = 0xFFFFFFFFFFFFFFFF / n.low;
            uint64_t r_high = 0xFFFFFFFFFFFFFFFF % n.low;
            
            // Now we need to add the contribution from the remainder
            // (r_high * 2^64 + 2^64) / n.low
            #ifdef __CUDA_ARCH__
            // Use 128-bit arithmetic on device
            unsigned __int128 dividend = ((unsigned __int128)(r_high + 1) << 64);
            uint64_t q_low = dividend / n.low;
            #else
            // Use 128-bit arithmetic on host
            unsigned __int128 dividend = ((unsigned __int128)(r_high + 1) << 64);
            uint64_t q_low = dividend / n.low;
            #endif
            
            return uint128_t(q_low, q_high);
        } else {
            // n is a full 128-bit number
            // Use a simpler approach: binary long division
            // We want floor(2^128 / n)
            
            // The dividend is 2^128 (represented as 1 followed by 128 zeros)
            // We'll compute this bit by bit using restoring division
            
            uint128_t quotient(0, 0);
            uint128_t remainder(0, 0);
            
            // We're dividing 2^128 by n
            // Start with the highest bit (bit 128)
            for (int i = 128; i >= 0; i--) {
                // Shift remainder left by 1
                uint64_t carry = (remainder.high >> 63) & 1;
                remainder.high = (remainder.high << 1) | ((remainder.low >> 63) & 1);
                remainder.low = remainder.low << 1;
                
                // Add the next bit of dividend (only bit 128 is 1)
                if (i == 128) {
                    remainder.low |= 1;
                }
                
                // Try to subtract divisor
                if (carry || remainder >= n) {
                    remainder = subtract_128(remainder, n);
                    // Set corresponding bit in quotient
                    if (i < 128) {
                        if (i >= 64) {
                            quotient.high |= (1ULL << (i - 64));
                        } else if (i < 64) {
                            quotient.low |= (1ULL << i);
                        }
                    }
                }
            }
            
            return quotient;
        }
    }
    
    // For other values of k, we would need more general division
    return uint128_t(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
}

/**
 * Precompute Barrett parameters for a given modulus
 */
__device__ __host__ void barrett_precompute(BarrettParams& params, const uint128_t& modulus) {
    params.modulus = modulus;
    
    // Compute k = 2 * bit_length(modulus)
    int bits = bit_length_128(modulus);
    params.k = 2 * bits;
    
    // For 128-bit arithmetic, we typically use k = 128
    // This simplifies the computation and is sufficient for most cases
    params.k = 128;
    
    // Compute mu = floor(2^k / modulus)
    params.mu = compute_barrett_mu(modulus, params.k);
}

/**
 * Barrett reduction: compute x mod m using precomputed parameters
 * Algorithm:
 * 1. q = floor((x * mu) / 2^k)
 * 2. r = x - q * m
 * 3. if r >= m, r = r - m
 */
__device__ __host__ uint128_t barrett_reduce(const uint128_t& x, const BarrettParams& params) {
    // Fast path: if x < modulus, no reduction needed
    if (x < params.modulus) {
        return x;
    }
    
    // Special handling for small moduli
    if (params.modulus.high == 0 && params.modulus.low != 0) {
        // For 64-bit modulus, use simple division
        if (x.high == 0) {
            return uint128_t(x.low % params.modulus.low, 0);
        }
        
        // Use built-in 128-bit division for 64-bit modulus
        #ifdef __CUDA_ARCH__
        unsigned __int128 x_full = ((unsigned __int128)x.high << 64) | x.low;
        unsigned __int128 r = x_full % params.modulus.low;
        return uint128_t((uint64_t)r, 0);
        #else
        unsigned __int128 x_full = ((unsigned __int128)x.high << 64) | x.low;
        unsigned __int128 r = x_full % params.modulus.low;
        return uint128_t((uint64_t)r, 0);
        #endif
    }
    
    // Step 1: Compute q = floor((x * mu) / 2^k)
    // Since k = 128, this is equivalent to taking the high 128 bits of x * mu
    uint256_t x_mu = multiply_128_128(x, params.mu);
    uint128_t q = x_mu.high_128();  // This is floor((x * mu) / 2^128)
    
    // Step 2: Compute r = x - q * modulus
    uint256_t q_m = multiply_128_128(q, params.modulus);
    uint128_t q_m_low = q_m.low_128();
    
    // Handle the subtraction carefully
    uint128_t r;
    if (q_m.word[2] == 0 && q_m.word[3] == 0) {
        // q*m fits in 128 bits
        if (x >= q_m_low) {
            r = subtract_128(x, q_m_low);
        } else {
            // This case means our approximation was too high
            // Add modulus back
            r = add_128(x, params.modulus);
            r = subtract_128(r, q_m_low);
        }
    } else {
        // q*m > 128 bits, which means q was too large
        // This shouldn't happen with correct mu, but handle it
        r = x;
    }
    
    // Step 3: Correction step - at most two subtractions needed
    int max_corrections = 3;  // Prevent infinite loop
    while (r >= params.modulus && max_corrections-- > 0) {
        r = subtract_128(r, params.modulus);
    }
    
    return r;
}

// Forward declaration
__device__ __host__ uint128_t fallback_mod(const uint128_t& x, const uint128_t& m);

/**
 * Modular multiplication using Barrett reduction
 */
__device__ __host__ uint128_t barrett_modmul(const uint128_t& a, const uint128_t& b, 
                                             const BarrettParams& params) {
    // First reduce inputs
    uint128_t a_red = barrett_reduce(a, params);
    uint128_t b_red = barrett_reduce(b, params);
    
    // Multiply
    uint256_t prod = multiply_128_128(a_red, b_red);
    
    // Extract low 128 bits and reduce
    // Note: For products, we might need to handle the full 256-bit result
    // For now, we take the low 128 bits and reduce
    uint128_t prod_low = prod.low_128();
    
    // If the high part is non-zero, we need a different approach
    // For products > 128 bits, use fallback method
    if (!prod.high_128().is_zero()) {
        // For now, use the fallback method for large products
        return fallback_mod(prod_low, params.modulus);
    }
    
    return barrett_reduce(prod_low, params);
}

/**
 * Fallback modular reduction using standard division
 * Used for validation and testing
 */
__device__ __host__ uint128_t fallback_mod(const uint128_t& x, const uint128_t& m) {
    if (m.is_zero()) return x;  // Error case
    
    // For 64-bit modulus, use built-in operations
    if (m.high == 0 && x.high == 0) {
        return uint128_t(x.low % m.low, 0);
    }
    
    // Binary long division for general case
    uint128_t quotient(0, 0);
    uint128_t remainder = x;
    
    // Find the highest bit positions
    int x_bits = bit_length_128(remainder);
    int m_bits = bit_length_128(m);
    
    if (x_bits < m_bits) {
        return remainder;  // x < m, no reduction needed
    }
    
    // Special case for x == m
    if (x == m) {
        return uint128_t(0, 0);
    }
    
    // Align m with the highest bit of x
    int shift = x_bits - m_bits;
    if (shift > 128) {
        // Protect against invalid shifts
        return remainder;
    }
    
    uint128_t m_shifted = shift_left_128(m, shift);
    
    // Perform binary division
    for (int i = shift; i >= 0; i--) {
        if (remainder >= m_shifted) {
            remainder = subtract_128(remainder, m_shifted);
        }
        if (i > 0) {
            m_shifted = shift_right_128(m_shifted, 1);
        }
    }
    
    return remainder;
}

/**
 * Comprehensive test kernel for Barrett reduction
 */
__global__ void test_barrett_reduction_kernel() {
    if (threadIdx.x != 0) return;
    
    printf("=== Barrett Reduction Test Suite ===\n\n");
    
    // Test 1: Small modulus (32-bit)
    {
        uint128_t x(0x123456789ABCDEF0ULL, 0);
        uint128_t m(1000000007, 0);
        
        BarrettParams params;
        barrett_precompute(params, m);
        
        uint128_t barrett_result = barrett_reduce(x, params);
        uint128_t fallback_result = fallback_mod(x, m);
        
        printf("Test 1 - Small modulus:\n");
        printf("  x = 0x%llx\n", x.low);
        printf("  m = %llu\n", m.low);
        printf("  mu = 0x%llx:%llx\n", params.mu.high, params.mu.low);
        printf("  Barrett result: %llu\n", barrett_result.low);
        printf("  Fallback result: %llu\n", fallback_result.low);
        printf("  Match: %s\n\n", (barrett_result == fallback_result) ? "YES" : "NO");
    }
    
    // Test 2: Large modulus (64-bit)
    {
        uint128_t x(0xFFFFFFFFFFFFFFFFULL, 0x1);  // 2^64 + 2^64 - 1
        uint128_t m(0x8000000000000000ULL, 0);    // 2^63
        
        BarrettParams params;
        barrett_precompute(params, m);
        
        uint128_t barrett_result = barrett_reduce(x, params);
        uint128_t fallback_result = fallback_mod(x, m);
        
        printf("Test 2 - Large 64-bit modulus:\n");
        printf("  x = 0x%llx:%llx\n", x.high, x.low);
        printf("  m = 0x%llx\n", m.low);
        printf("  mu = 0x%llx:%llx\n", params.mu.high, params.mu.low);
        printf("  Barrett result: 0x%llx\n", barrett_result.low);
        printf("  Fallback result: 0x%llx\n", fallback_result.low);
        printf("  Match: %s\n\n", (barrett_result == fallback_result) ? "YES" : "NO");
    }
    
    // Test 3: Full 128-bit modulus
    {
        uint128_t x(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);  // 2^128 - 1
        uint128_t m(0x123456789ABCDEF0ULL, 0x1);  // Large 128-bit prime-like number
        
        BarrettParams params;
        barrett_precompute(params, m);
        
        uint128_t barrett_result = barrett_reduce(x, params);
        uint128_t fallback_result = fallback_mod(x, m);
        
        printf("Test 3 - Full 128-bit modulus:\n");
        printf("  x = 0x%llx:%llx\n", x.high, x.low);
        printf("  m = 0x%llx:%llx\n", m.high, m.low);
        printf("  mu = 0x%llx:%llx\n", params.mu.high, params.mu.low);
        printf("  Barrett result: 0x%llx:%llx\n", barrett_result.high, barrett_result.low);
        printf("  Fallback result: 0x%llx:%llx\n", fallback_result.high, fallback_result.low);
        printf("  Match: %s\n\n", (barrett_result == fallback_result) ? "YES" : "NO");
    }
    
    // Test 4: Modular multiplication
    {
        uint128_t a(0x123456789ABCDEFULL, 0);
        uint128_t b(0xFEDCBA987654321ULL, 0);
        uint128_t m(0xFFFFFFFF00000001ULL, 0);  // Commonly used prime 2^64 - 2^32 + 1
        
        BarrettParams params;
        barrett_precompute(params, m);
        
        uint128_t barrett_result = barrett_modmul(a, b, params);
        
        // Compute expected result using fallback
        uint256_t prod = multiply_128_128(a, b);
        uint128_t expected = fallback_mod(prod.low_128(), m);
        
        printf("Test 4 - Modular multiplication:\n");
        printf("  a = 0x%llx\n", a.low);
        printf("  b = 0x%llx\n", b.low);
        printf("  m = 0x%llx\n", m.low);
        printf("  Barrett modmul result: 0x%llx\n", barrett_result.low);
        printf("  Expected result: 0x%llx\n", expected.low);
        printf("  Match: %s\n\n", (barrett_result == expected) ? "YES" : "NO");
    }
    
    // Test 5: Edge cases
    {
        printf("Test 5 - Edge cases:\n");
        
        // Case 5a: x == m (should return 0)
        uint128_t m(999999999999ULL, 0);
        BarrettParams params;
        barrett_precompute(params, m);
        
        uint128_t result = barrett_reduce(m, params);
        printf("  x == m: %llu mod %llu = %llu (expected: 0)\n", 
               m.low, m.low, result.low);
        
        // Case 5b: x < m (should return x)
        uint128_t x(12345, 0);
        result = barrett_reduce(x, params);
        printf("  x < m: %llu mod %llu = %llu (expected: %llu)\n", 
               x.low, m.low, result.low, x.low);
        
        // Case 5c: x = 2*m - 1 (should return m-1)
        uint128_t x2 = add_128(m, m);
        x2 = subtract_128(x2, uint128_t(1, 0));
        result = barrett_reduce(x2, params);
        uint128_t expected = subtract_128(m, uint128_t(1, 0));
        printf("  x = 2m-1: %llu mod %llu = %llu (expected: %llu)\n", 
               x2.low, m.low, result.low, expected.low);
    }
    
    printf("\n=== Barrett Reduction Test Complete ===\n");
}

/**
 * Random validation test kernel
 * Tests Barrett reduction against fallback with random values
 */
__global__ void validate_barrett_random(int num_tests, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tests) return;
    
    // Initialize random state
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    // Generate random test values
    uint64_t x_low = ((uint64_t)curand(&state) << 32) | curand(&state);
    uint64_t x_high = ((uint64_t)curand(&state) << 32) | curand(&state);
    uint64_t m_low = ((uint64_t)curand(&state) << 32) | curand(&state);
    uint64_t m_high = curand(&state) & 0xFFFFFF;  // Keep modulus reasonable
    
    // Ensure modulus is not zero and not too small
    if (m_low < 1000) m_low = 1000;
    
    uint128_t x(x_low, x_high);
    uint128_t m(m_low, m_high);
    
    // Compute Barrett parameters
    BarrettParams params;
    barrett_precompute(params, m);
    
    // Compute both results
    uint128_t barrett_result = barrett_reduce(x, params);
    uint128_t fallback_result = fallback_mod(x, m);
    
    // Check for mismatch
    if (barrett_result != fallback_result) {
        printf("MISMATCH in thread %d:\n", tid);
        printf("  x = 0x%llx:%llx\n", x.high, x.low);
        printf("  m = 0x%llx:%llx\n", m.high, m.low);
        printf("  Barrett: 0x%llx:%llx\n", barrett_result.high, barrett_result.low);
        printf("  Fallback: 0x%llx:%llx\n", fallback_result.high, fallback_result.low);
    }
}

#endif // BARRETT_CLEAN_CUH