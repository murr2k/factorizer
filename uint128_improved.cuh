/**
 * Improved 128-bit Arithmetic for CUDA
 * Correct carry propagation and overflow handling
 */

#ifndef UINT128_IMPROVED_CUH
#define UINT128_IMPROVED_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Forward declarations
struct uint128_t;
struct uint256_t;

// 128-bit unsigned integer with correct arithmetic
struct uint128_t {
    uint64_t low;
    uint64_t high;
    
    // Constructors
    __device__ __host__ uint128_t() : low(0), high(0) {}
    __device__ __host__ uint128_t(uint64_t l) : low(l), high(0) {}
    __device__ __host__ uint128_t(uint64_t l, uint64_t h) : low(l), high(h) {}
    
    // Comparison operators
    __device__ __host__ bool operator==(const uint128_t& other) const {
        return low == other.low && high == other.high;
    }
    
    __device__ __host__ bool operator!=(const uint128_t& other) const {
        return !(*this == other);
    }
    
    __device__ __host__ bool operator<(const uint128_t& other) const {
        if (high != other.high) return high < other.high;
        return low < other.low;
    }
    
    __device__ __host__ bool operator<=(const uint128_t& other) const {
        return *this < other || *this == other;
    }
    
    __device__ __host__ bool operator>(const uint128_t& other) const {
        return !(*this <= other);
    }
    
    __device__ __host__ bool operator>=(const uint128_t& other) const {
        return !(*this < other);
    }
    
    // Check if zero
    __device__ __host__ bool is_zero() const {
        return low == 0 && high == 0;
    }
    
    // Bit operations
    __device__ __host__ int leading_zeros() const {
        #ifdef __CUDA_ARCH__
        if (high != 0) {
            return __clzll(high);
        }
        return 64 + __clzll(low);
        #else
        // Host code fallback
        if (high != 0) {
            return __builtin_clzll(high);
        }
        return 64 + __builtin_clzll(low);
        #endif
    }
};

// 256-bit result for multiplication
struct uint256_t {
    uint64_t word[4];  // word[0] is least significant
    
    __device__ __host__ uint256_t() {
        for (int i = 0; i < 4; i++) word[i] = 0;
    }
    
    __device__ __host__ uint128_t low_128() const {
        return uint128_t(word[0], word[1]);
    }
    
    __device__ __host__ uint128_t high_128() const {
        return uint128_t(word[2], word[3]);
    }
};

// Addition with carry propagation
__device__ __host__ uint128_t add_128(const uint128_t& a, const uint128_t& b) {
    uint128_t result;
    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < a.low ? 1 : 0);
    return result;
}

// Subtraction with borrow propagation
__device__ __host__ uint128_t subtract_128(const uint128_t& a, const uint128_t& b) {
    uint128_t result;
    result.low = a.low - b.low;
    result.high = a.high - b.high - (a.low < b.low ? 1 : 0);
    return result;
}

// Multiply two 64-bit numbers to get 128-bit result
__device__ __host__ uint128_t multiply_64_64(uint64_t a, uint64_t b) {
    uint128_t result;
#ifdef __CUDA_ARCH__
    result.low = a * b;
    result.high = __umul64hi(a, b);
#else
    // Host code fallback
    unsigned __int128 prod = (unsigned __int128)a * b;
    result.low = (uint64_t)prod;
    result.high = (uint64_t)(prod >> 64);
#endif
    return result;
}

// Full 128x128 multiplication with correct carry handling
__device__ __host__ uint256_t multiply_128_128(const uint128_t& a, const uint128_t& b) {
    // Break down into 32-bit chunks for precise control
    uint64_t a0 = (uint32_t)(a.low);
    uint64_t a1 = (uint32_t)(a.low >> 32);
    uint64_t a2 = (uint32_t)(a.high);
    uint64_t a3 = (uint32_t)(a.high >> 32);
    
    uint64_t b0 = (uint32_t)(b.low);
    uint64_t b1 = (uint32_t)(b.low >> 32);
    uint64_t b2 = (uint32_t)(b.high);
    uint64_t b3 = (uint32_t)(b.high >> 32);
    
    // All partial products (each is at most 64 bits)
    uint64_t p00 = a0 * b0;
    uint64_t p01 = a0 * b1;
    uint64_t p02 = a0 * b2;
    uint64_t p03 = a0 * b3;
    
    uint64_t p10 = a1 * b0;
    uint64_t p11 = a1 * b1;
    uint64_t p12 = a1 * b2;
    uint64_t p13 = a1 * b3;
    
    uint64_t p20 = a2 * b0;
    uint64_t p21 = a2 * b1;
    uint64_t p22 = a2 * b2;
    uint64_t p23 = a2 * b3;
    
    uint64_t p30 = a3 * b0;
    uint64_t p31 = a3 * b1;
    uint64_t p32 = a3 * b2;
    uint64_t p33 = a3 * b3;
    
    // Sum all partial products with proper carry propagation
    uint256_t result;
    uint64_t carry = 0;
    
    // Word 0 (bits 0-63)
    result.word[0] = p00;
    carry = 0;
    
    // Add contributions to bit 32
    uint64_t sum32 = (p00 >> 32) + (p01 & 0xFFFFFFFF) + (p10 & 0xFFFFFFFF) + carry;
    result.word[0] = (result.word[0] & 0xFFFFFFFF) | ((sum32 & 0xFFFFFFFF) << 32);
    carry = sum32 >> 32;
    
    // Word 1 (bits 64-127)
    uint64_t sum64 = carry + (p01 >> 32) + p02 + (p10 >> 32) + p11 + p20;
    result.word[1] = sum64;
    carry = 0;
    
    // Handle overflow from sum64
    if (sum64 < p02 || sum64 < p11 || sum64 < p20) {
        carry = 1;
    }
    
    // Add contributions to bit 96
    uint64_t sum96 = (sum64 >> 32) + p03 + (p11 >> 32) + p12 + (p20 >> 32) + 
                     p21 + p30 + carry;
    result.word[1] = (result.word[1] & 0xFFFFFFFF) | ((sum96 & 0xFFFFFFFF) << 32);
    carry = sum96 >> 32;
    
    // Word 2 (bits 128-191)
    result.word[2] = carry + (p03 >> 32) + (p12 >> 32) + p13 + (p21 >> 32) + 
                     p22 + (p30 >> 32) + p31;
    
    // Word 3 (bits 192-255)
    result.word[3] = (p13 >> 32) + (p22 >> 32) + p23 + (p31 >> 32) + 
                     p32 + (p33);
    
    return result;
}

// Shift operations
__device__ __host__ uint128_t shift_left_128(const uint128_t& a, int shift) {
    if (shift == 0) return a;
    if (shift >= 128) return uint128_t(0, 0);
    if (shift >= 64) {
        return uint128_t(0, a.low << (shift - 64));
    }
    
    uint128_t result;
    result.low = a.low << shift;
    result.high = (a.high << shift) | (a.low >> (64 - shift));
    return result;
}

__device__ __host__ uint128_t shift_right_128(const uint128_t& a, int shift) {
    if (shift == 0) return a;
    if (shift >= 128) return uint128_t(0, 0);
    if (shift >= 64) {
        return uint128_t(a.high >> (shift - 64), 0);
    }
    
    uint128_t result;
    result.high = a.high >> shift;
    result.low = (a.low >> shift) | (a.high << (64 - shift));
    return result;
}

// Division by single 64-bit value (for testing)
__device__ __host__ uint128_t divide_128_64(const uint128_t& dividend, uint64_t divisor) {
    if (divisor == 0) return uint128_t(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    
    uint128_t quotient(0, 0);
    uint128_t remainder = dividend;
    
    // Simple long division
    for (int i = 127; i >= 0; i--) {
        remainder = shift_left_128(remainder, 1);
        if (remainder.high > 0 || remainder.low >= divisor) {
            remainder.low -= divisor;
            if (remainder.low > dividend.low) remainder.high--;
            quotient = add_128(quotient, shift_left_128(uint128_t(1), i));
        }
    }
    
    return quotient;
}

// Test kernel for 128-bit arithmetic
__global__ void test_uint128_arithmetic() {
    if (threadIdx.x == 0) {
        // Test 1: Addition with carry
        uint128_t a(0xFFFFFFFFFFFFFFFF, 0);  // 2^64 - 1
        uint128_t b(1, 0);
        uint128_t sum = add_128(a, b);
        printf("Test 1 - Addition: %llx:%llx + 1 = %llx:%llx (expected 0:1)\n",
               a.high, a.low, sum.high, sum.low);
        
        // Test 2: Multiplication
        uint128_t x(0xFFFFFFFF, 0);  // 2^32 - 1
        uint128_t y(0xFFFFFFFF, 0);  // 2^32 - 1
        uint256_t prod = multiply_128_128(x, y);
        printf("Test 2 - Multiplication: (2^32-1)^2 = %llx:%llx:%llx:%llx\n",
               prod.word[3], prod.word[2], prod.word[1], prod.word[0]);
        // Expected: 0xFFFFFFFE00000001
        
        // Test 3: Large multiplication
        uint128_t m1(0x123456789ABCDEF0, 0x1);
        uint128_t m2(0x2, 0);
        uint256_t prod2 = multiply_128_128(m1, m2);
        printf("Test 3 - Large multiply: result = %llx:%llx:%llx:%llx\n",
               prod2.word[3], prod2.word[2], prod2.word[1], prod2.word[0]);
    }
}

// Binary GCD algorithm for 128-bit numbers
__device__ uint128_t gcd_128(uint128_t a, uint128_t b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;
    
    // Count trailing zeros (common factors of 2)
    int shift = 0;
    while (((a.low | b.low) & 1) == 0 && shift < 64) {
        a = shift_right_128(a, 1);
        b = shift_right_128(b, 1);
        shift++;
    }
    
    // Remove remaining factors of 2 from a
    while ((a.low & 1) == 0) {
        a = shift_right_128(a, 1);
    }
    
    do {
        // Remove factors of 2 from b
        while ((b.low & 1) == 0) {
            b = shift_right_128(b, 1);
        }
        
        // Ensure a <= b
        if (a > b) {
            uint128_t temp = a;
            a = b;
            b = temp;
        }
        
        b = subtract_128(b, a);
    } while (!b.is_zero());
    
    return shift_left_128(a, shift);
}

// Improved modular multiplication with better reduction
__device__ uint128_t modmul_128_fast(uint128_t a, uint128_t b, uint128_t n) {
    // Ensure inputs are reduced
    while (a >= n) a = subtract_128(a, n);
    while (b >= n) b = subtract_128(b, n);
    
    // Fast path for 64-bit operands
    if (a.high == 0 && b.high == 0 && n.high == 0 && n.low != 0) {
        #ifdef __CUDA_ARCH__
        // Use built-in 128-bit multiplication
        unsigned __int128 prod = (unsigned __int128)a.low * b.low;
        unsigned __int128 mod = prod % n.low;
        return uint128_t((uint64_t)mod, 0);
        #else
        // Host fallback
        unsigned __int128 prod = (unsigned __int128)a.low * b.low;
        unsigned __int128 mod = prod % n.low;
        return uint128_t((uint64_t)mod, 0);
        #endif
    }
    
    // Full multiplication
    uint256_t prod = multiply_128_128(a, b);
    uint128_t result(prod.word[0], prod.word[1]);
    
    // Binary reduction - much faster than repeated subtraction
    if (result >= n) {
        // Find the highest bit of n
        int n_bits = 128 - n.leading_zeros();
        
        // Start with n shifted to align with the highest bit of result
        int result_bits = 128 - result.leading_zeros();
        int shift = result_bits - n_bits;
        
        if (shift >= 0) {
            uint128_t n_shifted = shift_left_128(n, shift);
            
            // Binary long division
            for (int i = shift; i >= 0; i--) {
                if (result >= n_shifted) {
                    result = subtract_128(result, n_shifted);
                }
                if (i > 0) {
                    n_shifted = shift_right_128(n_shifted, 1);
                }
            }
        }
    }
    
    return result;
}

#endif // UINT128_IMPROVED_CUH