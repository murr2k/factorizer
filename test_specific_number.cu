/**
 * Test factorization of specific number: 71123818302723020625487649
 * Using the v2.1 simple approach that we know works
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <chrono>

// Simple 128-bit type for this test
struct uint128_t {
    unsigned long long low;
    unsigned long long high;
    
    __host__ __device__ uint128_t() : low(0), high(0) {}
    __host__ __device__ uint128_t(unsigned long long l, unsigned long long h) : low(l), high(h) {}
};

// Parse decimal string to uint128
uint128_t parse_decimal(const char* str) {
    uint128_t result;
    result.low = 0;
    result.high = 0;
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            // Multiply by 10
            unsigned long long carry = 0;
            unsigned long long temp = result.low * 10ULL;
            carry = (result.low > temp / 10ULL) ? 1 : 0;
            result.low = temp;
            result.high = result.high * 10ULL + carry;
            
            // Add digit
            result.low += (str[i] - '0');
            if (result.low < (unsigned long long)(str[i] - '0')) {
                result.high++;
            }
        }
    }
    
    return result;
}

// Print uint128 in decimal (simplified)
void print_uint128_decimal(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
    } else {
        printf("(high: %llu, low: %llu)", n.high, n.low);
    }
}

// Optimized modular multiplication for 64-bit
__device__ unsigned long long modmul_optimized(
    unsigned long long a, 
    unsigned long long b, 
    unsigned long long n
) {
    unsigned long long res = 0;
    a %= n;
    
    while (b > 0) {
        if (b & 1) {
            res = (res + a) % n;
        }
        a = (a * 2) % n;
        b >>= 1;
    }
    
    return res;
}

// GCD function
__device__ unsigned long long gcd(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Pollard's Rho kernel for 64-bit numbers
__global__ void pollards_rho_64bit(
    unsigned long long n,
    unsigned long long* factor,
    int* found
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize PRNG
    curandState_t state;
    curand_init(clock64() + tid * 12345, tid, 0, &state);
    
    // Random starting values
    unsigned long long x = 2 + (curand(&state) % (n - 2));
    unsigned long long y = x;
    unsigned long long c = 1 + (curand(&state) % (n - 1));
    
    // Pollard's Rho algorithm
    for (int i = 0; i < 10000000 && !(*found); i++) {
        // f(x) = (x^2 + c) mod n
        x = modmul_optimized(x, x, n);
        x = (x + c) % n;
        
        // f(f(y))
        y = modmul_optimized(y, y, n);
        y = (y + c) % n;
        y = modmul_optimized(y, y, n);
        y = (y + c) % n;
        
        // Check GCD
        unsigned long long diff = (x > y) ? x - y : y - x;
        unsigned long long g = gcd(diff, n);
        
        if (g > 1 && g < n) {
            *factor = g;
            atomicExch(found, 1);
            return;
        }
        
        // Restart with new values periodically
        if (i % 100000 == 99999) {
            x = 2 + (curand(&state) % (n - 2));
            y = x;
            c = 1 + (curand(&state) % (n - 1));
        }
    }
}

int main() {
    printf("=== Factorization Test ===\n");
    printf("Number: 71123818302723020625487649\n\n");
    
    // Parse the number
    const char* number_str = "71123818302723020625487649";
    uint128_t full_number = parse_decimal(number_str);
    
    printf("Parsed number: ");
    print_uint128_decimal(full_number);
    printf("\n");
    
    // Check if it fits in 64 bits
    if (full_number.high == 0) {
        printf("Number fits in 64 bits!\n");
        printf("Value: %llu\n", full_number.low);
        
        // Factor using GPU
        unsigned long long* d_factor;
        int* d_found;
        cudaMalloc(&d_factor, sizeof(unsigned long long));
        cudaMalloc(&d_found, sizeof(int));
        cudaMemset(d_found, 0, sizeof(int));
        
        // Launch configuration
        int num_blocks = 32;
        int threads_per_block = 256;
        
        printf("\nLaunching %d blocks x %d threads\n", num_blocks, threads_per_block);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Launch kernel
        pollards_rho_64bit<<<num_blocks, threads_per_block>>>(
            full_number.low, d_factor, d_found
        );
        
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        // Get results
        int h_found;
        unsigned long long h_factor;
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_factor, d_factor, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        printf("\nTime: %.3f seconds\n", elapsed);
        
        if (h_found) {
            unsigned long long cofactor = full_number.low / h_factor;
            printf("✓ Factors found: %llu × %llu\n", h_factor, cofactor);
            
            // Verify
            if (h_factor * cofactor == full_number.low) {
                printf("✓ Verification: %llu × %llu = %llu\n", 
                       h_factor, cofactor, full_number.low);
            }
        } else {
            printf("✗ No factors found\n");
        }
        
        cudaFree(d_factor);
        cudaFree(d_found);
        
    } else {
        printf("Number is larger than 64 bits!\n");
        printf("High part: %llu\n", full_number.high);
        printf("Low part: %llu\n", full_number.low);
        printf("\nThis requires 128-bit arithmetic implementation.\n");
    }
    
    return 0;
}