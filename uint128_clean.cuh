#ifndef UINT128_CLEAN_CUH
#define UINT128_CLEAN_CUH

#include <cstdint>

struct uint128_t {
    uint64_t hi;
    uint64_t lo;
    
    // Default constructor
    __host__ __device__ uint128_t() : hi(0), lo(0) {}
    
    // Constructor from single uint64_t
    __host__ __device__ uint128_t(uint64_t value) : hi(0), lo(value) {}
    
    // Constructor from two uint64_t values
    __host__ __device__ uint128_t(uint64_t h, uint64_t l) : hi(h), lo(l) {}
    
    // Addition operator
    __host__ __device__ uint128_t operator+(const uint128_t& other) const {
        uint128_t result;
        result.lo = lo + other.lo;
        // Check for carry from low part
        result.hi = hi + other.hi + (result.lo < lo ? 1 : 0);
        return result;
    }
    
    // Subtraction operator
    __host__ __device__ uint128_t operator-(const uint128_t& other) const {
        uint128_t result;
        result.lo = lo - other.lo;
        // Check for borrow to low part
        result.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
        return result;
    }
    
    // Comparison operators
    __host__ __device__ bool operator<(const uint128_t& other) const {
        if (hi != other.hi) return hi < other.hi;
        return lo < other.lo;
    }
    
    __host__ __device__ bool operator>=(const uint128_t& other) const {
        return !(*this < other);
    }
    
    __host__ __device__ bool operator==(const uint128_t& other) const {
        return hi == other.hi && lo == other.lo;
    }
    
    __host__ __device__ bool operator!=(const uint128_t& other) const {
        return !(*this == other);
    }
    
    // Check if zero
    __host__ __device__ bool is_zero() const {
        return hi == 0 && lo == 0;
    }
    
    // Left shift by 1 bit
    __host__ __device__ uint128_t shift_left_1() const {
        uint128_t result;
        result.hi = (hi << 1) | (lo >> 63);
        result.lo = lo << 1;
        return result;
    }
    
    // Right shift by 1 bit
    __host__ __device__ uint128_t shift_right_1() const {
        uint128_t result;
        result.lo = (lo >> 1) | (hi << 63);
        result.hi = hi >> 1;
        return result;
    }
};

// Static multiplication function
// Multiplies two 64-bit numbers and returns 128-bit result
__host__ __device__ inline uint128_t mul(uint64_t a, uint64_t b) {
    // Split operands into 32-bit halves
    uint32_t a_lo = (uint32_t)a;
    uint32_t a_hi = (uint32_t)(a >> 32);
    uint32_t b_lo = (uint32_t)b;
    uint32_t b_hi = (uint32_t)(b >> 32);
    
    // Calculate partial products (each is at most 64 bits)
    uint64_t p00 = (uint64_t)a_lo * b_lo;
    uint64_t p01 = (uint64_t)a_lo * b_hi;
    uint64_t p10 = (uint64_t)a_hi * b_lo;
    uint64_t p11 = (uint64_t)a_hi * b_hi;
    
    // Assemble the 128-bit result
    // Low 32 bits come directly from p00
    uint64_t result_lo = p00;
    
    // Middle 64 bits come from p01 + p10 + (high 32 bits of p00)
    uint64_t middle = p01 + p10 + (p00 >> 32);
    
    // Add the low 32 bits of middle to result_lo
    result_lo = (result_lo & 0xFFFFFFFF) | (middle << 32);
    
    // High 64 bits come from p11 + (high 32 bits of middle)
    uint64_t result_hi = p11 + (middle >> 32);
    
    return uint128_t(result_hi, result_lo);
}

// Static division function
// Divides 128-bit number by 128-bit number, returns quotient
// Uses schoolbook long division algorithm
__host__ __device__ inline uint128_t div(const uint128_t& dividend, const uint128_t& divisor) {
    // Handle division by zero
    if (divisor.is_zero()) {
        return uint128_t(~0ULL, ~0ULL); // Return max value as error indicator
    }
    
    // Handle simple cases
    if (dividend < divisor) {
        return uint128_t(0);
    }
    
    if (dividend == divisor) {
        return uint128_t(1);
    }
    
    // Schoolbook long division
    uint128_t quotient(0);
    uint128_t remainder = dividend;
    
    // Find the highest bit position in divisor
    int divisor_bits = 0;
    uint128_t temp = divisor;
    while (temp.hi != 0 || temp.lo != 0) {
        divisor_bits++;
        temp = temp.shift_right_1();
    }
    
    // Find the highest bit position in dividend
    int dividend_bits = 0;
    temp = dividend;
    while (temp.hi != 0 || temp.lo != 0) {
        dividend_bits++;
        temp = temp.shift_right_1();
    }
    
    // Shift divisor left to align with dividend
    int shift = dividend_bits - divisor_bits;
    uint128_t shifted_divisor = divisor;
    for (int i = 0; i < shift; i++) {
        shifted_divisor = shifted_divisor.shift_left_1();
    }
    
    // Perform long division
    for (int i = 0; i <= shift; i++) {
        quotient = quotient.shift_left_1();
        if (remainder >= shifted_divisor) {
            remainder = remainder - shifted_divisor;
            quotient.lo |= 1;
        }
        shifted_divisor = shifted_divisor.shift_right_1();
    }
    
    return quotient;
}

// Helper function to get the remainder from division
__host__ __device__ inline uint128_t mod(const uint128_t& dividend, const uint128_t& divisor) {
    // Handle division by zero
    if (divisor.is_zero()) {
        return dividend;
    }
    
    // Handle simple cases
    if (dividend < divisor) {
        return dividend;
    }
    
    if (dividend == divisor) {
        return uint128_t(0);
    }
    
    // Use the same algorithm as div but return remainder
    uint128_t remainder = dividend;
    
    // Find the highest bit position in divisor
    int divisor_bits = 0;
    uint128_t temp = divisor;
    while (temp.hi != 0 || temp.lo != 0) {
        divisor_bits++;
        temp = temp.shift_right_1();
    }
    
    // Find the highest bit position in dividend
    int dividend_bits = 0;
    temp = dividend;
    while (temp.hi != 0 || temp.lo != 0) {
        dividend_bits++;
        temp = temp.shift_right_1();
    }
    
    // Shift divisor left to align with dividend
    int shift = dividend_bits - divisor_bits;
    uint128_t shifted_divisor = divisor;
    for (int i = 0; i < shift; i++) {
        shifted_divisor = shifted_divisor.shift_left_1();
    }
    
    // Perform long division
    for (int i = 0; i <= shift; i++) {
        if (remainder >= shifted_divisor) {
            remainder = remainder - shifted_divisor;
        }
        shifted_divisor = shifted_divisor.shift_right_1();
    }
    
    return remainder;
}

#endif // UINT128_CLEAN_CUH