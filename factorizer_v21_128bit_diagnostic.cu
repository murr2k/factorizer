/**
 * v2.1 Factorizer for 128-bit numbers - DIAGNOSTIC VERSION
 * Handles up to 39 decimal digits
 * 
 * This version includes extensive diagnostics to understand what's happening
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include "uint128_improved.cuh"

// Parse decimal string to uint128
uint128_t parse_decimal(const char* str) {
    uint128_t result(0, 0);
    uint128_t ten(10, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            // result = result * 10 + digit
            uint256_t temp = multiply_128_128(result, ten);
            result = uint128_t(temp.word[0], temp.word[1]);
            result = add_128(result, uint128_t(str[i] - '0', 0));
        }
    }
    
    return result;
}

// Print uint128 in decimal
void print_uint128(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
        return;
    }
    
    // For larger numbers, just show hex for now
    printf("0x%llx%016llx", n.high, n.low);
}

// Use the improved modular multiplication from the header
__device__ inline uint128_t modmul_128_improved(uint128_t a, uint128_t b, uint128_t n) {
    return modmul_128_fast(a, b, n);
}

// Optimized modular multiplication for 128-bit (original for comparison)
__device__ uint128_t modmul_128(uint128_t a, uint128_t b, uint128_t n) {
    // Simple method: ensure a and b are reduced first
    if (a >= n) {
        // Reduce a mod n (simplified)
        while (a >= n) {
            a = subtract_128(a, n);
        }
    }
    if (b >= n) {
        while (b >= n) {
            b = subtract_128(b, n);
        }
    }
    
    // Multiply and reduce
    uint256_t prod = multiply_128_128(a, b);
    uint128_t result(prod.word[0], prod.word[1]);
    
    // Simple reduction
    while (result >= n) {
        result = subtract_128(result, n);
    }
    
    return result;
}

// Structure to track algorithm state
struct AlgorithmState {
    uint128_t x;
    uint128_t y;
    uint128_t last_gcd;
    int iterations;
    int reduction_count;
    bool found_cycle;
    bool found_factor;
};

__global__ void pollards_rho_128bit_diagnostic(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    AlgorithmState* states,
    int max_iterations = 10000000,  // Reduced for diagnostics
    bool use_improved_modmul = true
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize variables
    uint128_t x(2 + tid * 7919, 0);  // Use larger prime for better distribution
    uint128_t y = x;
    uint128_t c(1 + (tid * 1009) % 10000, 0);  // Much larger range for c
    uint128_t factor(1, 0);
    
    // Initialize state tracking
    AlgorithmState& state = states[tid];
    state.x = x;
    state.y = y;
    state.last_gcd = uint128_t(0, 0);
    state.iterations = 0;
    state.reduction_count = 0;
    state.found_cycle = false;
    state.found_factor = false;
    
    // Track values for cycle detection
    uint128_t saved_x = x;
    int saved_iteration = 0;
    
    // Print initial state for thread 0
    if (tid == 0 && threadIdx.x == 0) {
        printf("\n[Thread 0] Starting with:\n");
        printf("  x = 0x%llx:%llx\n", x.high, x.low);
        printf("  c = %llu\n", c.low);
        printf("  n = 0x%llx:%llx\n", n.high, n.low);
        printf("  Using %s modmul\n\n", use_improved_modmul ? "improved" : "original");
    }
    
    for (int i = 0; i < max_iterations; i++) {
        // x = (x^2 + c) mod n
        if (use_improved_modmul) {
            x = modmul_128_improved(x, x, n);
        } else {
            x = modmul_128(x, x, n);
        }
        x = add_128(x, c);
        if (x >= n) {
            x = subtract_128(x, n);
            state.reduction_count++;
        }
        
        // y = f(f(y))
        if (use_improved_modmul) {
            y = modmul_128_improved(y, y, n);
        } else {
            y = modmul_128(y, y, n);
        }
        y = add_128(y, c);
        if (y >= n) {
            y = subtract_128(y, n);
            state.reduction_count++;
        }
        
        if (use_improved_modmul) {
            y = modmul_128_improved(y, y, n);
        } else {
            y = modmul_128(y, y, n);
        }
        y = add_128(y, c);
        if (y >= n) {
            y = subtract_128(y, n);
            state.reduction_count++;
        }
        
        // Calculate |x - y|
        uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
        
        // Calculate GCD
        factor = gcd_128(diff, n);
        state.last_gcd = factor;
        
        // Diagnostic output every 100k iterations for thread 0
        if (tid == 0 && i > 0 && i % 100000 == 0) {
            printf("[Thread 0] Iteration %d:\n", i);
            printf("  x = 0x%llx:%llx\n", x.high, x.low);
            printf("  y = 0x%llx:%llx\n", y.high, y.low);
            printf("  diff = 0x%llx:%llx\n", diff.high, diff.low);
            printf("  gcd = 0x%llx:%llx\n", factor.high, factor.low);
            printf("  reductions = %d\n\n", state.reduction_count);
        }
        
        if (factor.low > 1 && factor < n) {
            // Found a factor!
            int idx = atomicAdd(factor_count, 1);
            if (idx == 0) {
                factors[0] = factor;
                // Try to get cofactor
                if (n.high == 0 && factor.high == 0 && factor.low != 0) {
                    factors[1] = uint128_t(n.low / factor.low, 0);
                }
                
                // Report finding
                printf("\n[Thread %d] FOUND FACTOR at iteration %d!\n", tid, i);
                printf("  Factor = 0x%llx:%llx\n", factor.high, factor.low);
            }
            state.found_factor = true;
            state.iterations = i;
            break;
        }
        
        // Cycle detection
        if (i > 0 && i % 1000000 == 0) {
            if (x == saved_x) {
                if (tid == 0) {
                    printf("[Thread %d] CYCLE DETECTED at iteration %d!\n", tid, i);
                    printf("  Cycle length = %d\n", i - saved_iteration);
                }
                state.found_cycle = true;
                
                // Re-randomize more aggressively
                c = uint128_t((c.low * 7919 + tid * 1009 + i) % 1000000 + 1, 0);
                x = uint128_t((x.low + c.low) % n.low, 0);
                y = x;
                
                if (tid == 0) {
                    printf("  Re-randomized with c = %llu\n\n", c.low);
                }
            }
            saved_x = x;
            saved_iteration = i;
        }
        
        // Warp cooperation
        if (i % 10000 == 0) {
            unsigned mask = __ballot_sync(0xFFFFFFFF, factor.low > 1);
            if (mask != 0) break;
        }
        
        // Re-randomize periodically with better randomization
        if (i % (500000 + tid * 10000) == 0 && i > 0) {
            uint64_t old_c = c.low;
            c = uint128_t((c.low * 6364136223846793005ULL + tid + i) % 1000000 + 1, 0);
            if (tid == 0 && i % 1000000 == 0) {
                printf("[Thread 0] Re-randomized: c changed from %llu to %llu\n", old_c, c.low);
            }
        }
        
        state.iterations = i;
        state.x = x;
        state.y = y;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <number> [improved_modmul=1]\n", argv[0]);
        printf("  improved_modmul: 0 = use original, 1 = use improved (default)\n");
        return 1;
    }
    
    bool use_improved = true;
    if (argc >= 3) {
        use_improved = (atoi(argv[2]) != 0);
    }
    
    // Parse input
    uint128_t n = parse_decimal(argv[1]);
    
    printf("=== v2.1 128-bit Factorizer - DIAGNOSTIC MODE ===\n");
    printf("Input: %s\n", argv[1]);
    printf("Number: ");
    print_uint128(n);
    printf("\n");
    printf("Using: %s modular multiplication\n", use_improved ? "Improved" : "Original");
    
    // Check if it's odd (for Montgomery)
    printf("Optimization potential: %s\n\n", (n.low & 1) ? "Montgomery-capable" : "Standard");
    
    // Allocate device memory
    uint128_t* d_factors;
    int* d_factor_count;
    AlgorithmState* d_states;
    
    cudaMalloc(&d_factors, 2 * sizeof(uint128_t));
    cudaMalloc(&d_factor_count, sizeof(int));
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    // Configure grid - fewer threads for diagnostic mode
    int blocks = 32;
    int threads = 128;
    int total_threads = blocks * threads;
    
    cudaMalloc(&d_states, total_threads * sizeof(AlgorithmState));
    
    printf("Launching %d blocks x %d threads = %d total threads\n", 
           blocks, threads, total_threads);
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    pollards_rho_128bit_diagnostic<<<blocks, threads>>>(
        n, d_factors, d_factor_count, d_states, 10000000, use_improved);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // Get timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get results
    int h_factor_count;
    uint128_t h_factors[2];
    AlgorithmState* h_states = new AlgorithmState[total_threads];
    
    cudaMemcpy(&h_factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_factors, d_factors, 2 * sizeof(uint128_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_states, d_states, total_threads * sizeof(AlgorithmState), cudaMemcpyDeviceToHost);
    
    printf("\nTime: %.3f seconds\n", duration.count() / 1000.0);
    
    // Print summary of thread states
    int cycles_found = 0;
    int max_iterations = 0;
    int total_reductions = 0;
    
    for (int i = 0; i < total_threads; i++) {
        if (h_states[i].found_cycle) cycles_found++;
        if (h_states[i].iterations > max_iterations) max_iterations = h_states[i].iterations;
        total_reductions += h_states[i].reduction_count;
    }
    
    printf("\n=== Algorithm Statistics ===\n");
    printf("Max iterations by any thread: %d\n", max_iterations);
    printf("Threads that found cycles: %d / %d\n", cycles_found, total_threads);
    printf("Total modular reductions: %d\n", total_reductions);
    printf("Avg reductions per thread: %.2f\n", (float)total_reductions / total_threads);
    
    // Sample some thread final states
    printf("\n=== Sample Thread States ===\n");
    for (int i = 0; i < min(5, total_threads); i++) {
        printf("Thread %d: iterations=%d, found_factor=%s, found_cycle=%s\n",
               i, h_states[i].iterations, 
               h_states[i].found_factor ? "yes" : "no",
               h_states[i].found_cycle ? "yes" : "no");
        printf("  Final x: 0x%llx:%llx\n", h_states[i].x.high, h_states[i].x.low);
        printf("  Last GCD: 0x%llx:%llx\n", h_states[i].last_gcd.high, h_states[i].last_gcd.low);
    }
    
    printf("\n=== Results ===\n");
    if (h_factor_count > 0 && h_factors[0].low > 1) {
        printf("✓ Factors found:\n");
        printf("  Factor 1: ");
        print_uint128(h_factors[0]);
        printf("\n");
        
        if (h_factors[1].low > 1) {
            printf("  Factor 2: ");
            print_uint128(h_factors[1]);
            printf("\n");
            
            // Verify if both are 64-bit
            if (h_factors[0].high == 0 && h_factors[1].high == 0) {
                printf("\nVerification: %llu × %llu = %llu\n", 
                       h_factors[0].low, h_factors[1].low, 
                       h_factors[0].low * h_factors[1].low);
            }
        }
    } else {
        printf("✗ No factors found\n");
        printf("\nPossible issues:\n");
        printf("- Number might be prime\n");
        printf("- Algorithm needs more iterations\n");
        printf("- Modular arithmetic overflow issues\n");
        if (cycles_found > 0) {
            printf("- Many threads found cycles (bad randomization)\n");
        }
    }
    
    // Cleanup
    delete[] h_states;
    cudaFree(d_factors);
    cudaFree(d_factor_count);
    cudaFree(d_states);
    
    return h_factor_count > 0 ? 0 : 1;
}