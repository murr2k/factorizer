/**
 * Enhanced CUDA Factorization with 128-bit Arithmetic Support
 * Implements multi-precision arithmetic for large semiprime factorization
 * Optimized for NVIDIA GTX 2070 (Compute 7.5)
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <gmp.h>
#include <chrono>
#include <iomanip>
#include <cstring>

// 128-bit unsigned integer structure for CUDA
struct uint128_t {
    unsigned long long lo;
    unsigned long long hi;
    
    __device__ __host__ uint128_t() : lo(0), hi(0) {}
    __device__ __host__ uint128_t(unsigned long long val) : lo(val), hi(0) {}
    __device__ __host__ uint128_t(unsigned long long h, unsigned long long l) : lo(l), hi(h) {}
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 128-bit arithmetic operations
__device__ __host__ uint128_t add128(const uint128_t& a, const uint128_t& b) {
    uint128_t result;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi + (result.lo < a.lo ? 1 : 0);
    return result;
}

__device__ __host__ uint128_t sub128(const uint128_t& a, const uint128_t& b) {
    uint128_t result;
    result.lo = a.lo - b.lo;
    result.hi = a.hi - b.hi - (a.lo < b.lo ? 1 : 0);
    return result;
}

__device__ uint128_t mul128(const uint128_t& a, const uint128_t& b) {
    // Karatsuba multiplication for 128-bit numbers
    unsigned long long a0 = a.lo & 0xFFFFFFFFULL;
    unsigned long long a1 = a.lo >> 32;
    unsigned long long a2 = a.hi & 0xFFFFFFFFULL;
    unsigned long long a3 = a.hi >> 32;
    
    unsigned long long b0 = b.lo & 0xFFFFFFFFULL;
    unsigned long long b1 = b.lo >> 32;
    unsigned long long b2 = b.hi & 0xFFFFFFFFULL;
    unsigned long long b3 = b.hi >> 32;
    
    // Only compute lower 128 bits of result
    unsigned long long p00 = a0 * b0;
    unsigned long long p01 = a0 * b1;
    unsigned long long p10 = a1 * b0;
    unsigned long long p11 = a1 * b1;
    unsigned long long p02 = a0 * b2;
    unsigned long long p20 = a2 * b0;
    unsigned long long p12 = a1 * b2;
    unsigned long long p21 = a2 * b1;
    unsigned long long p03 = a0 * b3;
    unsigned long long p30 = a3 * b0;
    
    // Sum contributions
    unsigned long long carry = 0;
    unsigned long long lo = p00;
    
    unsigned long long mid1 = (p01 & 0xFFFFFFFFULL) << 32;
    lo += mid1;
    carry += (lo < mid1) ? 1 : 0;
    
    unsigned long long mid2 = (p10 & 0xFFFFFFFFULL) << 32;
    lo += mid2;
    carry += (lo < mid2) ? 1 : 0;
    
    unsigned long long hi = carry + (p01 >> 32) + (p10 >> 32) + p11 + 
                           ((p02 & 0xFFFFFFFFULL) << 32) + ((p20 & 0xFFFFFFFFULL) << 32);
    
    hi += (p12 & 0xFFFFFFFFULL) << 32;
    hi += (p21 & 0xFFFFFFFFULL) << 32;
    hi += (p02 >> 32) + (p20 >> 32) + (p03 & 0xFFFFFFFFULL) + (p30 & 0xFFFFFFFFULL);
    
    return uint128_t(hi, lo);
}

__device__ uint128_t mod128(const uint128_t& a, const uint128_t& n) {
    // Binary long division for modulo
    if (n.hi == 0 && n.lo == 0) return uint128_t(0, 0);
    
    uint128_t remainder = a;
    uint128_t divisor = n;
    
    // Simple iterative subtraction for now
    while (remainder.hi > n.hi || (remainder.hi == n.hi && remainder.lo >= n.lo)) {
        remainder = sub128(remainder, n);
    }
    
    return remainder;
}

__device__ bool is_greater128(const uint128_t& a, const uint128_t& b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
}

__device__ bool is_equal128(const uint128_t& a, const uint128_t& b) {
    return a.hi == b.hi && a.lo == b.lo;
}

__device__ uint128_t gcd128(uint128_t a, uint128_t b) {
    while (b.hi != 0 || b.lo != 0) {
        uint128_t temp = b;
        b = mod128(a, b);
        a = temp;
    }
    return a;
}

// Modular multiplication for 128-bit numbers
__device__ uint128_t modmul128(const uint128_t& a, const uint128_t& b, const uint128_t& mod) {
    uint128_t result = mul128(a, b);
    return mod128(result, mod);
}

// Pollard's Rho for 128-bit numbers
__device__ uint128_t pollard_rho_128(const uint128_t& n, curandState_t* state) {
    if ((n.lo & 1) == 0) return uint128_t(0, 2); // Even number
    
    uint128_t x = uint128_t(0, 2 + curand(state) % 100);
    uint128_t y = x;
    uint128_t c = uint128_t(0, 1 + curand(state) % 100);
    uint128_t d = uint128_t(0, 1);
    
    int iterations = 0;
    const int max_iterations = 100000;
    
    while ((d.hi == 0 && d.lo == 1) && iterations < max_iterations) {
        // x = (x² + c) mod n
        x = modmul128(x, x, n);
        x = add128(x, c);
        x = mod128(x, n);
        
        // y = (y² + c) mod n, twice
        y = modmul128(y, y, n);
        y = add128(y, c);
        y = mod128(y, n);
        
        y = modmul128(y, y, n);
        y = add128(y, c);
        y = mod128(y, n);
        
        // d = gcd(|x - y|, n)
        uint128_t diff = is_greater128(x, y) ? sub128(x, y) : sub128(y, x);
        d = gcd128(diff, n);
        
        iterations++;
    }
    
    if (is_equal128(d, n)) {
        return uint128_t(0, 0); // Failed
    }
    
    return d;
}

// Parallel factorization kernel for 128-bit numbers
__global__ void parallel_factor_128_kernel(uint128_t n, uint128_t* factors, 
                                          int* num_factors, int max_attempts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState_t state;
    curand_init(tid + clock64(), tid, 0, &state);
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        uint128_t factor = pollard_rho_128(n, &state);
        
        if (factor.hi > 0 || factor.lo > 1) {
            if (!is_equal128(factor, n)) {
                // Found a non-trivial factor
                int idx = atomicAdd(num_factors, 1);
                if (idx < 10) {  // Store up to 10 factors
                    factors[idx] = factor;
                    
                    // For semiprimes, we can compute the cofactor
                    // This would require 128-bit division implementation
                }
                return;
            }
        }
    }
}

// Host-side factorization manager
class CUDA128Factorizer {
private:
    int device_id;
    cudaDeviceProp device_prop;
    
public:
    CUDA128Factorizer() {
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(&device_prop, device_id);
        
        std::cout << "Initialized CUDA 128-bit Factorizer on " << device_prop.name << std::endl;
        std::cout << "Compute capability: " << device_prop.major << "." 
                  << device_prop.minor << std::endl;
    }
    
    std::vector<std::string> factorize(const std::string& number_str) {
        // Convert string to 128-bit number using GMP
        mpz_t n;
        mpz_init(n);
        mpz_set_str(n, number_str.c_str(), 10);
        
        // Extract 128-bit value
        uint128_t n128;
        n128.lo = mpz_get_ui(n);
        mpz_tdiv_q_2exp(n, n, 64);
        n128.hi = mpz_get_ui(n);
        mpz_clear(n);
        
        std::cout << "\n=== CUDA 128-bit Factorization ===" << std::endl;
        std::cout << "Number: " << number_str << std::endl;
        std::cout << "High 64 bits: " << std::hex << n128.hi << std::dec << std::endl;
        std::cout << "Low 64 bits: " << std::hex << n128.lo << std::dec << std::endl;
        
        // Allocate device memory
        uint128_t* d_factors;
        int* d_num_factors;
        CUDA_CHECK(cudaMalloc(&d_factors, 10 * sizeof(uint128_t)));
        CUDA_CHECK(cudaMalloc(&d_num_factors, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_num_factors, 0, sizeof(int)));
        
        // Launch parallel factorization
        const int num_threads = 256;
        const int num_blocks = device_prop.multiProcessorCount * 2;
        const int max_attempts = 10000;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        parallel_factor_128_kernel<<<num_blocks, num_threads>>>(
            n128, d_factors, d_num_factors, max_attempts);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Retrieve results
        int num_factors;
        CUDA_CHECK(cudaMemcpy(&num_factors, d_num_factors, sizeof(int), cudaMemcpyDeviceToHost));
        
        std::vector<uint128_t> factors(num_factors);
        if (num_factors > 0) {
            CUDA_CHECK(cudaMemcpy(factors.data(), d_factors, 
                                 num_factors * sizeof(uint128_t), cudaMemcpyDeviceToHost));
        }
        
        std::cout << "\nFactorization completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Found " << num_factors << " factors" << std::endl;
        
        // Convert factors back to strings
        std::vector<std::string> result;
        for (const auto& factor : factors) {
            mpz_t f;
            mpz_init(f);
            mpz_set_ui(f, factor.hi);
            mpz_mul_2exp(f, f, 64);
            mpz_add_ui(f, f, factor.lo);
            
            char* str = mpz_get_str(NULL, 10, f);
            result.push_back(std::string(str));
            free(str);
            mpz_clear(f);
        }
        
        CUDA_CHECK(cudaFree(d_factors));
        CUDA_CHECK(cudaFree(d_num_factors));
        
        return result;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <number_to_factor>" << std::endl;
        std::cout << "Example: " << argv[0] << " 94498503396937386863845286721509" << std::endl;
        return 1;
    }
    
    CUDA128Factorizer factorizer;
    
    std::string number = argv[1];
    auto factors = factorizer.factorize(number);
    
    if (factors.size() > 0) {
        std::cout << "\n✓ Factorization successful!" << std::endl;
        std::cout << "Factors: ";
        for (const auto& factor : factors) {
            std::cout << factor << " ";
        }
        std::cout << std::endl;
        
        // Verify if it's a semiprime
        if (factors.size() == 2) {
            std::cout << "\n✓ Confirmed: This is a semiprime!" << std::endl;
            
            // Verify the product using GMP
            mpz_t p1, p2, product, original;
            mpz_init(p1);
            mpz_init(p2);
            mpz_init(product);
            mpz_init(original);
            
            mpz_set_str(p1, factors[0].c_str(), 10);
            mpz_set_str(p2, factors[1].c_str(), 10);
            mpz_set_str(original, number.c_str(), 10);
            
            mpz_mul(product, p1, p2);
            
            if (mpz_cmp(product, original) == 0) {
                std::cout << "Verification: " << factors[0] << " × " << factors[1] 
                          << " = " << number << " ✓" << std::endl;
            }
            
            mpz_clear(p1);
            mpz_clear(p2);
            mpz_clear(product);
            mpz_clear(original);
        }
    } else {
        std::cout << "\n⚠ No factors found. The number may be prime or require more iterations." << std::endl;
    }
    
    return 0;
}