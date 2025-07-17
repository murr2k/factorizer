/**
 * Algorithm Selector Header
 * Interface for intelligent factorization algorithm selection
 */

#ifndef ALGORITHM_SELECTOR_CUH
#define ALGORITHM_SELECTOR_CUH

#include <cuda_runtime.h>
#include "uint128_improved.cuh"

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

// Progress estimation
struct ProgressEstimate {
    float completion_percentage;
    int estimated_remaining_ms;
    const char* current_phase;
};

// Core functions
__host__ NumberAnalysis analyze_number(const uint128_t& n);
__host__ AlgorithmChoice select_algorithm(const NumberAnalysis& analysis);
__host__ ProgressEstimate estimate_progress(
    const AlgorithmChoice& choice,
    const NumberAnalysis& analysis,
    int elapsed_ms,
    int iterations_completed
);
__host__ bool should_switch_algorithm(
    const AlgorithmChoice& current,
    const NumberAnalysis& analysis,
    int elapsed_ms,
    int iterations_without_progress
);

// Utility functions
__host__ __device__ int get_bit_length(const uint128_t& n);
__host__ bool detect_perfect_power(const uint128_t& n, int& base, int& exponent);
__host__ bool miller_rabin_test(const uint128_t& n, int rounds = 20);
__host__ float estimate_smoothness(const uint128_t& n, const uint64_t* small_factors, int factor_count);
__host__ uint64_t get_available_gpu_memory();

// GPU kernels
__global__ void quick_trial_division(
    uint128_t n,
    uint64_t* small_factors,
    int* factor_count,
    int max_trial
);

#endif // ALGORITHM_SELECTOR_CUH