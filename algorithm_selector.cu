/**
 * Algorithm Selector for CUDA Factorizer
 * Intelligent selection of optimal factorization algorithms based on input characteristics
 * 
 * Algorithms available:
 * 1. Trial Division - for small factors
 * 2. Pollard's Rho - for medium composite numbers
 * 3. Quadratic Sieve - for larger semiprimes (future)
 * 4. ECM - for numbers with medium-sized factors (future)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <cmath>
#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include "memory_optimizer.cuh"

// Algorithm types
enum FactorizationAlgorithm {
    ALGO_TRIAL_DIVISION = 0,
    ALGO_POLLARD_RHO = 1,
    ALGO_QUADRATIC_SIEVE = 2,
    ALGO_ECM = 3,
    ALGO_COMBINED = 4
};

// Algorithm selection result
struct AlgorithmChoice {
    FactorizationAlgorithm primary_algorithm;
    FactorizationAlgorithm fallback_algorithm;
    int confidence_score;  // 0-100
    int estimated_time_ms;
    int trial_division_limit;
    int pollard_rho_iterations;
    bool use_gpu;
    int num_threads;
    int num_blocks;
};

// Number characteristics
struct NumberAnalysis {
    uint128_t n;
    int bit_length;
    int decimal_digits;
    bool is_even;
    int small_factor_count;
    uint64_t smallest_factor;
    bool likely_prime;
    bool perfect_power;
    int power_base;
    int power_exponent;
    float smoothness_estimate;
    uint64_t available_gpu_memory;
};

// Fast bit length calculation
__host__ __device__ int get_bit_length(const uint128_t& n) {
    if (n.high == 0) {
        return 64 - __builtin_clzll(n.low);
    } else {
        return 128 - __builtin_clzll(n.high);
    }
}

// Count decimal digits
__host__ int count_decimal_digits(const uint128_t& n) {
    if (n.high == 0) {
        if (n.low == 0) return 1;
        return (int)floor(log10((double)n.low)) + 1;
    }
    // Approximate for large numbers
    int bits = get_bit_length(n);
    return (int)(bits * 0.30103) + 1;  // log10(2) ≈ 0.30103
}

// Perfect power detection
__host__ bool detect_perfect_power(const uint128_t& n, int& base, int& exponent) {
    // Check for perfect squares first
    if (n.high == 0 && n.low < (1ULL << 32)) {
        uint64_t sqrt_n = (uint64_t)sqrt((double)n.low);
        if (sqrt_n * sqrt_n == n.low) {
            base = sqrt_n;
            exponent = 2;
            return true;
        }
    }
    
    // Check higher powers up to reasonable limit
    int max_exp = get_bit_length(n);
    for (int k = 3; k <= min(max_exp, 64); k++) {
        // Binary search for k-th root
        uint64_t low = 1, high = (1ULL << (64 / k));
        
        while (low <= high) {
            uint64_t mid = (low + high) / 2;
            
            // Calculate mid^k
            uint128_t power(1, 0);
            for (int i = 0; i < k; i++) {
                uint256_t temp = multiply_128_128(power, uint128_t(mid, 0));
                if (temp.word[2] != 0 || temp.word[3] != 0) {
                    // Overflow
                    high = mid - 1;
                    break;
                }
                power = uint128_t(temp.word[0], temp.word[1]);
                
                if (power > n) {
                    high = mid - 1;
                    break;
                }
            }
            
            if (power == n) {
                base = mid;
                exponent = k;
                return true;
            } else if (power < n) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
    
    return false;
}

// Miller-Rabin primality test (probabilistic)
__host__ bool miller_rabin_test(const uint128_t& n, int rounds = 20) {
    if (n.high == 0) {
        if (n.low < 2) return false;
        if (n.low == 2 || n.low == 3) return true;
        if (n.low % 2 == 0) return false;
        
        // For small numbers, use deterministic test
        if (n.low < 1000000) {
            for (uint64_t i = 3; i * i <= n.low; i += 2) {
                if (n.low % i == 0) return false;
            }
            return true;
        }
    }
    
    // Find r and d such that n-1 = 2^r * d
    uint128_t n_minus_1 = subtract_128(n, uint128_t(1, 0));
    uint128_t d = n_minus_1;
    int r = 0;
    
    while ((d.low & 1) == 0) {
        d = shift_right_128(d, 1);
        r++;
    }
    
    // Perform rounds of testing
    for (int i = 0; i < rounds; i++) {
        // Random base between 2 and n-2
        uint64_t a_val = 2 + (rand() % min(n.low - 4, 1000000ULL));
        uint128_t a(a_val, 0);
        
        // Compute a^d mod n
        uint128_t x = modpow_128(a, d, n);
        
        if (x == uint128_t(1, 0) || x == n_minus_1) {
            continue;
        }
        
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = modmul_128(x, x, n);
            if (x == n_minus_1) {
                composite = false;
                break;
            }
        }
        
        if (composite) return false;
    }
    
    return true;
}

// Quick trial division to find small factors
__global__ void quick_trial_division(
    uint128_t n,
    uint64_t* small_factors,
    int* factor_count,
    int max_trial
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Small primes table
    __shared__ uint64_t primes[168];
    if (threadIdx.x < 168) {
        const uint64_t small_primes[168] = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
            157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
            239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
            331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
            421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
            509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
            613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
            709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
            821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
            919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
        };
        primes[threadIdx.x] = small_primes[threadIdx.x];
    }
    __syncthreads();
    
    // Check divisibility by small primes
    for (int i = tid; i < min(168, max_trial); i += stride) {
        uint64_t p = primes[i];
        
        // Only check if n could be divisible by p
        if (n.high == 0 && n.low % p == 0) {
            int idx = atomicAdd(factor_count, 1);
            if (idx < 10) {  // Store up to 10 small factors
                small_factors[idx] = p;
            }
        }
    }
}

// Smoothness estimation heuristic
__host__ float estimate_smoothness(const uint128_t& n, const uint64_t* small_factors, int factor_count) {
    if (factor_count == 0) {
        return 0.0f;  // No small factors found
    }
    
    // Calculate product of found factors
    uint128_t factor_product(1, 0);
    for (int i = 0; i < factor_count; i++) {
        factor_product = multiply_128_64(factor_product, small_factors[i]);
    }
    
    // Estimate based on how much the small factors reduce n
    int original_bits = get_bit_length(n);
    int reduced_bits = get_bit_length(factor_product);
    
    return (float)reduced_bits / original_bits;
}

// Get available GPU memory
__host__ uint64_t get_available_gpu_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return (uint64_t)free_mem;
}

// Main analysis function
__host__ NumberAnalysis analyze_number(const uint128_t& n) {
    NumberAnalysis analysis;
    analysis.n = n;
    analysis.bit_length = get_bit_length(n);
    analysis.decimal_digits = count_decimal_digits(n);
    analysis.is_even = (n.low & 1) == 0;
    analysis.small_factor_count = 0;
    analysis.smallest_factor = 0;
    analysis.likely_prime = false;
    analysis.perfect_power = false;
    analysis.power_base = 0;
    analysis.power_exponent = 0;
    analysis.smoothness_estimate = 0.0f;
    analysis.available_gpu_memory = get_available_gpu_memory();
    
    // Check for perfect power
    int base, exponent;
    if (detect_perfect_power(n, base, exponent)) {
        analysis.perfect_power = true;
        analysis.power_base = base;
        analysis.power_exponent = exponent;
        printf("Perfect power detected: %d^%d\n", base, exponent);
        return analysis;
    }
    
    // Quick primality test for small numbers
    if (analysis.bit_length <= 64) {
        analysis.likely_prime = miller_rabin_test(n, 10);
        if (analysis.likely_prime) {
            printf("Number is likely prime\n");
            return analysis;
        }
    }
    
    // GPU trial division for small factors
    uint64_t* d_small_factors;
    int* d_factor_count;
    cudaMalloc(&d_small_factors, 10 * sizeof(uint64_t));
    cudaMalloc(&d_factor_count, sizeof(int));
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    quick_trial_division<<<8, 256>>>(n, d_small_factors, d_factor_count, 1000);
    cudaDeviceSynchronize();
    
    // Get results
    uint64_t h_small_factors[10];
    cudaMemcpy(&analysis.small_factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (analysis.small_factor_count > 0) {
        cudaMemcpy(h_small_factors, d_small_factors, 
                   min(10, analysis.small_factor_count) * sizeof(uint64_t), 
                   cudaMemcpyDeviceToHost);
        analysis.smallest_factor = h_small_factors[0];
        analysis.smoothness_estimate = estimate_smoothness(n, h_small_factors, analysis.small_factor_count);
    }
    
    cudaFree(d_small_factors);
    cudaFree(d_factor_count);
    
    return analysis;
}

// Algorithm selection heuristics
__host__ AlgorithmChoice select_algorithm(const NumberAnalysis& analysis) {
    AlgorithmChoice choice;
    choice.confidence_score = 50;  // Default confidence
    choice.use_gpu = true;
    choice.num_blocks = 16;
    choice.num_threads = 256;
    choice.trial_division_limit = 10000;
    choice.pollard_rho_iterations = 1000000;
    
    // Case 1: Perfect power
    if (analysis.perfect_power) {
        printf("Strategy: Factor the base %d\n", analysis.power_base);
        choice.primary_algorithm = ALGO_TRIAL_DIVISION;
        choice.fallback_algorithm = ALGO_POLLARD_RHO;
        choice.confidence_score = 95;
        choice.estimated_time_ms = 10;
        return choice;
    }
    
    // Case 2: Likely prime
    if (analysis.likely_prime) {
        printf("Strategy: Number is likely prime, use extended primality testing\n");
        choice.primary_algorithm = ALGO_TRIAL_DIVISION;
        choice.fallback_algorithm = ALGO_POLLARD_RHO;
        choice.confidence_score = 20;  // Low confidence in factoring
        choice.estimated_time_ms = 5000;
        choice.pollard_rho_iterations = 10000000;  // More iterations needed
        return choice;
    }
    
    // Case 3: Small number (< 64 bits)
    if (analysis.bit_length <= 64) {
        printf("Strategy: Small number, use trial division followed by Pollard's rho\n");
        choice.primary_algorithm = ALGO_TRIAL_DIVISION;
        choice.fallback_algorithm = ALGO_POLLARD_RHO;
        choice.trial_division_limit = min(1000000ULL, (uint64_t)sqrt((double)analysis.n.low));
        choice.confidence_score = 90;
        choice.estimated_time_ms = 100;
        return choice;
    }
    
    // Case 4: Smooth number (many small factors)
    if (analysis.smoothness_estimate > 0.3f) {
        printf("Strategy: Smooth number detected, extensive trial division\n");
        choice.primary_algorithm = ALGO_TRIAL_DIVISION;
        choice.fallback_algorithm = ALGO_POLLARD_RHO;
        choice.trial_division_limit = 100000;
        choice.confidence_score = 85;
        choice.estimated_time_ms = 500;
        return choice;
    }
    
    // Case 5: Medium number (64-96 bits)
    if (analysis.bit_length <= 96) {
        printf("Strategy: Medium number, Pollard's rho with optimized parameters\n");
        choice.primary_algorithm = ALGO_POLLARD_RHO;
        choice.fallback_algorithm = ALGO_ECM;
        choice.trial_division_limit = 50000;
        choice.pollard_rho_iterations = 5000000;
        choice.num_blocks = 32;
        choice.confidence_score = 75;
        choice.estimated_time_ms = 2000;
        return choice;
    }
    
    // Case 6: Large number (> 96 bits)
    if (analysis.bit_length > 96) {
        printf("Strategy: Large number, combined approach\n");
        choice.primary_algorithm = ALGO_COMBINED;
        choice.fallback_algorithm = ALGO_QUADRATIC_SIEVE;
        
        // Adjust parameters based on available GPU memory
        if (analysis.available_gpu_memory > 4ULL * 1024 * 1024 * 1024) {  // > 4GB
            choice.num_blocks = 64;
            choice.num_threads = 512;
            choice.pollard_rho_iterations = 20000000;
            printf("  Using high GPU memory configuration\n");
        } else {
            choice.num_blocks = 32;
            choice.num_threads = 256;
            choice.pollard_rho_iterations = 10000000;
            printf("  Using standard GPU memory configuration\n");
        }
        
        choice.confidence_score = 60;
        choice.estimated_time_ms = 10000;
        return choice;
    }
    
    // Default case
    choice.primary_algorithm = ALGO_POLLARD_RHO;
    choice.fallback_algorithm = ALGO_TRIAL_DIVISION;
    return choice;
}

// Progress estimation based on algorithm and number size
struct ProgressEstimate {
    float completion_percentage;
    int estimated_remaining_ms;
    const char* current_phase;
};

__host__ ProgressEstimate estimate_progress(
    const AlgorithmChoice& choice,
    const NumberAnalysis& analysis,
    int elapsed_ms,
    int iterations_completed
) {
    ProgressEstimate estimate;
    
    switch (choice.primary_algorithm) {
        case ALGO_TRIAL_DIVISION: {
            int total_primes = choice.trial_division_limit / 10;  // Approximate
            int checked = iterations_completed;
            estimate.completion_percentage = (float)checked / total_primes * 100;
            estimate.current_phase = "Trial division";
            break;
        }
        
        case ALGO_POLLARD_RHO: {
            // Pollard's rho expected iterations: O(sqrt(p)) where p is smallest factor
            // Use heuristic based on number size
            float expected_iterations = pow(2.0, analysis.bit_length / 4.0);
            estimate.completion_percentage = min(99.0f, (float)iterations_completed / expected_iterations * 100);
            estimate.current_phase = "Pollard's rho iteration";
            break;
        }
        
        case ALGO_COMBINED: {
            // Combined approach has multiple phases
            if (elapsed_ms < 1000) {
                estimate.current_phase = "Initial trial division";
                estimate.completion_percentage = elapsed_ms / 10.0f;
            } else if (elapsed_ms < 5000) {
                estimate.current_phase = "Pollard's rho phase 1";
                estimate.completion_percentage = 10 + (elapsed_ms - 1000) / 40.0f;
            } else {
                estimate.current_phase = "Pollard's rho phase 2";
                estimate.completion_percentage = 20 + min(79.0f, (elapsed_ms - 5000) / 100.0f);
            }
            break;
        }
        
        default:
            estimate.completion_percentage = 0;
            estimate.current_phase = "Unknown";
    }
    
    // Estimate remaining time
    if (estimate.completion_percentage > 0) {
        float rate = estimate.completion_percentage / elapsed_ms;
        estimate.estimated_remaining_ms = (int)((100 - estimate.completion_percentage) / rate);
    } else {
        estimate.estimated_remaining_ms = choice.estimated_time_ms;
    }
    
    return estimate;
}

// Dynamic algorithm switching based on runtime performance
__host__ bool should_switch_algorithm(
    const AlgorithmChoice& current,
    const NumberAnalysis& analysis,
    int elapsed_ms,
    int iterations_without_progress
) {
    // Switch if no progress for too long
    if (iterations_without_progress > current.pollard_rho_iterations / 10) {
        printf("No progress detected, switching algorithm\n");
        return true;
    }
    
    // Switch if taking much longer than estimated
    if (elapsed_ms > current.estimated_time_ms * 3) {
        printf("Exceeded time estimate by 3x, switching algorithm\n");
        return true;
    }
    
    // Switch if GPU memory pressure detected
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem < total_mem / 10) {  // Less than 10% free
        printf("Low GPU memory detected, switching to more memory-efficient algorithm\n");
        return true;
    }
    
    return false;
}

// Example usage and testing
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }
    
    printf("=== CUDA Factorizer Algorithm Selector ===\n\n");
    
    // Parse input number
    uint128_t n = parse_decimal(argv[1]);
    printf("Input number: %s\n", argv[1]);
    printf("Hex representation: 0x%llx:%llx\n\n", n.high, n.low);
    
    // Analyze number characteristics
    printf("=== Number Analysis ===\n");
    auto start_analysis = std::chrono::high_resolution_clock::now();
    NumberAnalysis analysis = analyze_number(n);
    auto end_analysis = std::chrono::high_resolution_clock::now();
    auto analysis_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_analysis - start_analysis);
    
    printf("Bit length: %d\n", analysis.bit_length);
    printf("Decimal digits: %d\n", analysis.decimal_digits);
    printf("Even: %s\n", analysis.is_even ? "Yes" : "No");
    printf("Small factors found: %d\n", analysis.small_factor_count);
    if (analysis.small_factor_count > 0) {
        printf("Smallest factor: %llu\n", analysis.smallest_factor);
        printf("Smoothness estimate: %.2f%%\n", analysis.smoothness_estimate * 100);
    }
    printf("Available GPU memory: %.2f GB\n", analysis.available_gpu_memory / (1024.0 * 1024 * 1024));
    printf("Analysis time: %lld ms\n\n", analysis_time.count());
    
    // Select algorithm
    printf("=== Algorithm Selection ===\n");
    AlgorithmChoice choice = select_algorithm(analysis);
    
    const char* algo_names[] = {
        "Trial Division", "Pollard's Rho", "Quadratic Sieve", "ECM", "Combined"
    };
    
    printf("Primary algorithm: %s\n", algo_names[choice.primary_algorithm]);
    printf("Fallback algorithm: %s\n", algo_names[choice.fallback_algorithm]);
    printf("Confidence score: %d%%\n", choice.confidence_score);
    printf("Estimated time: %d ms\n", choice.estimated_time_ms);
    printf("GPU configuration: %d blocks x %d threads\n", choice.num_blocks, choice.num_threads);
    
    if (choice.primary_algorithm == ALGO_TRIAL_DIVISION || 
        choice.primary_algorithm == ALGO_COMBINED) {
        printf("Trial division limit: %d\n", choice.trial_division_limit);
    }
    
    if (choice.primary_algorithm == ALGO_POLLARD_RHO || 
        choice.primary_algorithm == ALGO_COMBINED) {
        printf("Pollard's rho iterations: %d\n", choice.pollard_rho_iterations);
    }
    
    printf("\n=== Recommendation Summary ===\n");
    printf("The selected algorithm should find factors with %d%% confidence\n", choice.confidence_score);
    printf("Expected completion time: ~%d seconds\n", choice.estimated_time_ms / 1000);
    
    if (choice.confidence_score < 50) {
        printf("\nWARNING: Low confidence score. The number may be:\n");
        printf("- A large prime\n");
        printf("- A product of two large primes (RSA-like)\n");
        printf("- Requiring specialized algorithms not yet implemented\n");
    }
    
    return 0;
}