/**
 * Intelligent CUDA Factorizer
 * Uses algorithm selection system to choose optimal factorization method
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <thread>
#include <atomic>
#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include "algorithm_selector.cuh"

// Global status for multi-algorithm coordination
struct FactorizationStatus {
    std::atomic<bool> factor_found;
    std::atomic<int> iterations_completed;
    std::atomic<int> iterations_without_progress;
    uint128_t factor;
    std::chrono::high_resolution_clock::time_point start_time;
};

// Enhanced trial division with early termination
__global__ void trial_division_enhanced(
    uint128_t n,
    uint128_t* device_factor,
    volatile int* device_status,
    int start_prime,
    int end_prime
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Generate primes on the fly using simple sieve
    for (uint64_t p = start_prime + tid * 2 + 1; p < end_prime && *device_status == 0; p += stride * 2) {
        // Skip even numbers except 2
        if (p > 2 && (p & 1) == 0) continue;
        
        // Simple primality check for p
        bool is_prime = true;
        for (uint64_t i = 3; i * i <= p && i < 1000; i += 2) {
            if (p % i == 0) {
                is_prime = false;
                break;
            }
        }
        
        if (!is_prime && p > 2) continue;
        
        // Check if p divides n
        if (n.high == 0) {
            if (n.low % p == 0) {
                int expected = 0;
                if (atomicCAS((int*)device_status, expected, 1) == expected) {
                    *device_factor = uint128_t(p, 0);
                    printf("Trial division: Found factor %llu\n", p);
                }
                return;
            }
        } else {
            // For large n, use modular arithmetic
            uint128_t p_128(p, 0);
            uint128_t remainder = mod_128(n, p_128);
            if (remainder.is_zero()) {
                int expected = 0;
                if (atomicCAS((int*)device_status, expected, 1) == expected) {
                    *device_factor = p_128;
                    printf("Trial division: Found factor %llu\n", p);
                }
                return;
            }
        }
    }
}

// Adaptive Pollard's Rho with parameter optimization
__global__ void pollard_rho_adaptive(
    uint128_t n,
    uint128_t* device_factor,
    volatile int* device_status,
    int max_iterations,
    int* iteration_counter,
    int restart_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random number generator
    curandState state;
    curand_init(clock64() + tid * 997 + blockIdx.x * 13, tid, 0, &state);
    
    // Barrett reduction setup
    Barrett128 barrett;
    barrett.n = n;
    barrett.precompute();
    
    int local_iterations = 0;
    int restarts = 0;
    
    while (local_iterations < max_iterations && *device_status == 0) {
        // Adaptive starting values
        uint64_t seed = curand(&state);
        uint128_t x((seed % n.low) | 2, 0);  // Ensure not 0 or 1
        uint128_t y = x;
        uint128_t c((seed >> 32) % 100 + 1, 0);
        
        // Brent's improvement parameters
        uint128_t ys = x;
        uint128_t q(1, 0);
        int r = 1;
        int m = 128;
        
        bool restart_needed = false;
        
        for (int i = 0; i < restart_threshold && !restart_needed && *device_status == 0; i++) {
            // Brent's cycle finding
            if (i == r) {
                ys = y;
                r *= 2;
                q = uint128_t(1, 0);
            }
            
            // Batch GCD computation
            int batch_size = min(m, r - i);
            for (int j = 0; j < batch_size; j++) {
                // y = (y^2 + c) mod n
                y = barrett.mod_square(y);
                y = add_128(y, c);
                if (y >= n) y = subtract_128(y, n);
                
                // Accumulate differences
                uint128_t diff = (x >= y) ? subtract_128(x, y) : subtract_128(y, x);
                q = barrett.mod_mul(q, diff);
                
                // Check for zero product (failure)
                if (q.is_zero()) {
                    restart_needed = true;
                    break;
                }
            }
            
            // Compute GCD
            uint128_t g = gcd_128(q, n);
            
            // Check if we found a factor
            if (!g.is_zero() && g != uint128_t(1, 0) && g != n) {
                int expected = 0;
                if (atomicCAS((int*)device_status, expected, 1) == expected) {
                    *device_factor = g;
                    printf("Pollard's Rho: Found factor after %d iterations (thread %d)\n", 
                           local_iterations + i, tid);
                }
                return;
            }
            
            // Move tortoise
            if (i % 100 == 0) {
                x = y;
            }
            
            local_iterations++;
            atomicAdd(iteration_counter, 1);
        }
        
        restarts++;
        
        // Adaptive restart with different parameters
        if (restarts % 10 == 0 && tid == 0) {
            printf("Thread %d: %d restarts, trying different approach\n", tid, restarts);
        }
    }
}

// Combined factorization with intelligent switching
bool factorize_intelligent(const char* number_str, bool verbose = true) {
    // Parse input
    uint128_t n = parse_decimal(number_str);
    
    if (verbose) {
        printf("\n=== Intelligent CUDA Factorizer ===\n");
        printf("Target: %s\n", number_str);
        printf("Binary: %d bits\n\n", get_bit_length(n));
    }
    
    // Analyze number
    auto analysis_start = std::chrono::high_resolution_clock::now();
    NumberAnalysis analysis = analyze_number(n);
    auto analysis_end = std::chrono::high_resolution_clock::now();
    auto analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_end - analysis_start);
    
    if (verbose) {
        printf("Analysis completed in %lld ms\n", analysis_time.count());
    }
    
    // Handle special cases
    if (analysis.perfect_power) {
        printf("\n✓ Perfect power detected: %d^%d\n", analysis.power_base, analysis.power_exponent);
        printf("Factorization: %d^%d = %s\n", analysis.power_base, analysis.power_exponent, number_str);
        return true;
    }
    
    if (analysis.likely_prime) {
        printf("\n✗ Number is likely prime (confidence > 99.9%%)\n");
        return false;
    }
    
    // Select algorithm
    AlgorithmChoice choice = select_algorithm(analysis);
    
    if (verbose) {
        printf("\nSelected strategy: ");
        switch (choice.primary_algorithm) {
            case ALGO_TRIAL_DIVISION:
                printf("Trial Division (limit: %d)\n", choice.trial_division_limit);
                break;
            case ALGO_POLLARD_RHO:
                printf("Pollard's Rho (%d iterations)\n", choice.pollard_rho_iterations);
                break;
            case ALGO_COMBINED:
                printf("Combined approach\n");
                break;
            default:
                printf("Custom algorithm\n");
        }
        printf("Confidence: %d%%\n", choice.confidence_score);
        printf("Estimated time: %d seconds\n\n", choice.estimated_time_ms / 1000);
    }
    
    // Allocate device memory
    uint128_t* device_factor;
    int* device_status;
    int* device_iterations;
    cudaMalloc(&device_factor, sizeof(uint128_t));
    cudaMalloc(&device_status, sizeof(int));
    cudaMalloc(&device_iterations, sizeof(int));
    cudaMemset(device_status, 0, sizeof(int));
    cudaMemset(device_iterations, 0, sizeof(int));
    
    // Start factorization
    auto start_time = std::chrono::high_resolution_clock::now();
    bool factor_found = false;
    uint128_t factor;
    
    // Phase 1: Quick trial division if appropriate
    if (choice.primary_algorithm == ALGO_TRIAL_DIVISION || 
        choice.primary_algorithm == ALGO_COMBINED) {
        
        if (verbose) printf("Phase 1: Trial division...\n");
        
        trial_division_enhanced<<<8, 256>>>(
            n, device_factor, device_status, 2, choice.trial_division_limit
        );
        
        cudaDeviceSynchronize();
        
        // Check status
        int status;
        cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (status == 1) {
            cudaMemcpy(&factor, device_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
            factor_found = true;
        }
    }
    
    // Phase 2: Pollard's Rho if needed
    if (!factor_found && 
        (choice.primary_algorithm == ALGO_POLLARD_RHO || 
         choice.primary_algorithm == ALGO_COMBINED)) {
        
        if (verbose) printf("Phase 2: Pollard's Rho...\n");
        
        // Reset status
        cudaMemset(device_status, 0, sizeof(int));
        
        // Launch adaptive Pollard's Rho
        pollard_rho_adaptive<<<choice.num_blocks, choice.num_threads>>>(
            n, device_factor, device_status, 
            choice.pollard_rho_iterations,
            device_iterations,
            10000  // restart threshold
        );
        
        // Monitor progress
        int last_iterations = 0;
        int no_progress_count = 0;
        
        while (!factor_found) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            int status;
            cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (status == 1) {
                cudaMemcpy(&factor, device_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
                factor_found = true;
                break;
            }
            
            // Check progress
            int current_iterations;
            cudaMemcpy(&current_iterations, device_iterations, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (current_iterations == last_iterations) {
                no_progress_count++;
            } else {
                no_progress_count = 0;
            }
            last_iterations = current_iterations;
            
            // Calculate elapsed time
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            
            // Progress estimation
            if (verbose && elapsed.count() % 1000 < 100) {
                ProgressEstimate progress = estimate_progress(
                    choice, analysis, elapsed.count(), current_iterations
                );
                printf("\rProgress: %.1f%% | Phase: %s | ETA: %d s    ",
                       progress.completion_percentage,
                       progress.current_phase,
                       progress.estimated_remaining_ms / 1000);
                fflush(stdout);
            }
            
            // Check if we should switch algorithms
            if (should_switch_algorithm(choice, analysis, elapsed.count(), no_progress_count)) {
                if (verbose) printf("\n\nSwitching to fallback algorithm...\n");
                break;
            }
            
            // Timeout check
            if (elapsed.count() > 300000) {  // 5 minutes
                if (verbose) printf("\n\nTimeout reached.\n");
                break;
            }
        }
        
        cudaDeviceSynchronize();
    }
    
    // Get final timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Results
    if (verbose) printf("\n\n");
    
    if (factor_found) {
        printf("=== SUCCESS ===\n");
        printf("Factor found: ");
        print_uint128_decimal(factor);
        printf("\n");
        
        // Calculate and verify cofactor
        uint128_t cofactor = divide_128(n, factor);
        printf("Cofactor: ");
        print_uint128_decimal(cofactor);
        printf("\n");
        
        // Verify
        uint256_t product = multiply_128_128(factor, cofactor);
        uint128_t check = uint128_t(product.word[0], product.word[1]);
        
        if (check == n) {
            printf("✓ Verification successful\n");
            
            // Check if cofactor is prime
            if (miller_rabin_test(cofactor, 20)) {
                printf("✓ Cofactor is prime\n");
            } else {
                printf("⚠ Cofactor is composite (further factorization possible)\n");
            }
        } else {
            printf("✗ Verification failed\n");
        }
        
        printf("\nTotal time: %.3f seconds\n", duration.count() / 1000.0);
    } else {
        printf("=== NO FACTOR FOUND ===\n");
        printf("The number might be:\n");
        printf("- Prime\n");
        printf("- A product of two large primes\n");
        printf("- Requiring more advanced algorithms\n");
        printf("\nTotal time: %.3f seconds\n", duration.count() / 1000.0);
    }
    
    // Cleanup
    cudaFree(device_factor);
    cudaFree(device_status);
    cudaFree(device_iterations);
    
    return factor_found;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <number> [--quiet]\n", argv[0]);
        printf("Example: %s 15482526220500967432610341\n", argv[0]);
        return 1;
    }
    
    bool verbose = true;
    if (argc > 2 && strcmp(argv[2], "--quiet") == 0) {
        verbose = false;
    }
    
    // Initialize CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (verbose) {
        printf("Using GPU: %s (%.1f GB memory)\n", prop.name, prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    }
    
    // Factorize
    bool success = factorize_intelligent(argv[1], verbose);
    
    return success ? 0 : 1;
}