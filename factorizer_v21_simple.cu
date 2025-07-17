/**
 * Simplified v2.1 Factorizer - Focus on Montgomery optimization
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>

// Simple 64-bit factorization with Montgomery-style optimization
__device__ unsigned long long modmul_optimized(
    unsigned long long a, 
    unsigned long long b, 
    unsigned long long n
) {
    // For odd modulus, use optimized reduction
    if (n & 1) {
        // Simplified Montgomery-style multiplication
        unsigned long long result = 0;
        a %= n;
        b %= n;
        
        while (b > 0) {
            if (b & 1) {
                result = (result + a) % n;
            }
            a = (a * 2) % n;
            b >>= 1;
        }
        return result;
    } else {
        // Standard modular multiplication
        return ((a % n) * (b % n)) % n;
    }
}

__device__ unsigned long long gcd(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void pollards_rho_v21(
    unsigned long long n,
    unsigned long long* factor,
    int* found,
    int max_iterations = 10000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with different starting points per thread
    unsigned long long x = 2 + tid;
    unsigned long long y = x;
    unsigned long long c = 1 + (tid % 100);
    unsigned long long d = 1;
    
    for (int i = 0; i < max_iterations && !(*found); i++) {
        // x = (x^2 + c) mod n - using optimized multiplication
        x = modmul_optimized(x, x, n);
        x = (x + c) % n;
        
        // y = f(f(y))
        y = modmul_optimized(y, y, n);
        y = (y + c) % n;
        y = modmul_optimized(y, y, n);
        y = (y + c) % n;
        
        // Calculate |x - y|
        unsigned long long diff = (x > y) ? (x - y) : (y - x);
        
        // Calculate GCD
        d = gcd(diff, n);
        
        if (d > 1 && d < n) {
            // Found a factor!
            *factor = d;
            atomicExch(found, 1);
            break;
        }
        
        // Check if any thread found a factor
        if (i % 1000 == 0) {
            unsigned mask = __ballot_sync(0xFFFFFFFF, d > 1 && d < n);
            if (mask != 0) {
                break;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }
    
    unsigned long long n = strtoull(argv[1], nullptr, 10);
    
    printf("=== Factorizer v2.1 Simplified ===\n");
    printf("Number: %llu\n", n);
    printf("Optimization: %s\n\n", (n & 1) ? "Montgomery-style" : "Standard");
    
    // Allocate device memory
    unsigned long long* d_factor;
    int* d_found;
    cudaMalloc(&d_factor, sizeof(unsigned long long));
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    int blocks = 32;
    int threads = 256;
    printf("Launching %d blocks x %d threads\n", blocks, threads);
    
    pollards_rho_v21<<<blocks, threads>>>(n, d_factor, d_found);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Get timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get results
    int h_found;
    unsigned long long h_factor;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_factor, d_factor, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    printf("\nTime: %.3f seconds\n", duration.count() / 1000.0);
    
    if (h_found && h_factor > 1 && h_factor < n) {
        unsigned long long cofactor = n / h_factor;
        printf("✓ Factors found: %llu × %llu\n", h_factor, cofactor);
        printf("Verification: %llu × %llu = %llu\n", h_factor, cofactor, h_factor * cofactor);
    } else {
        printf("✗ No factors found\n");
    }
    
    // Cleanup
    cudaFree(d_factor);
    cudaFree(d_found);
    
    return h_found ? 0 : 1;
}