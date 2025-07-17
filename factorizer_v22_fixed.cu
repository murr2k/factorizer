/**
 * CUDA Factorizer Version 2.2.0 - Unified Integration Master (Fixed)
 * Complete factorization solution with intelligent algorithm selection
 * 
 * Features:
 * - Unified API for all factorization methods
 * - Intelligent algorithm selection based on number characteristics
 * - Smooth transitions between algorithms with fallback mechanisms
 * - Real-time progress reporting with GPU metrics
 * - Comprehensive error handling and recovery
 * - Optimized for 26-digit test case
 * 
 * Copyright (c) 2025 - Integration Master Edition
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
#include <mutex>
#include <condition_variable>
#include <vector>
#include <algorithm>

// Include all v2.x components
#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include "montgomery_reduction.cuh"
#include "progress_monitor_fixed.cuh"

// Version information
#define VERSION_MAJOR 2
#define VERSION_MINOR 2
#define VERSION_PATCH 0
#define VERSION_STRING "2.2.0"

// Algorithm types
enum class AlgorithmType {
    AUTO_SELECT,
    TRIAL_DIVISION,
    POLLARDS_RHO_BASIC,
    POLLARDS_RHO_BRENT,
    POLLARDS_RHO_PARALLEL,
    QUADRATIC_SIEVE,
    ELLIPTIC_CURVE
};

// Algorithm state
enum class AlgorithmState {
    NOT_STARTED,
    RUNNING,
    COMPLETED,
    FAILED,
    TIMEOUT,
    TRANSITIONING
};

// Unified result structure
struct FactorizationResult {
    uint128_t factors[64];
    int factor_count;
    AlgorithmType algorithm_used;
    int total_iterations;
    double total_time_ms;
    int gpu_threads_used;
    bool is_complete;
    char error_message[256];
    
    // Performance metrics
    double iterations_per_second;
    double gpu_utilization_avg;
    int algorithm_switches;
};

// Algorithm configuration
struct AlgorithmConfig {
    AlgorithmType type;
    int max_iterations;
    int timeout_ms;
    int num_blocks;
    int threads_per_block;
    bool use_barrett;
    bool use_montgomery;
    bool use_memory_optimization;
    
    // Algorithm-specific parameters
    union {
        struct {
            int batch_size;
            int cycle_detection_interval;
        } pollards_rho;
        
        struct {
            int sieve_size;
            int smooth_bound;
        } quadratic_sieve;
    };
};

// Global configuration
struct FactorizerConfig {
    bool verbose;
    bool show_progress;
    bool auto_algorithm;
    bool gpu_monitoring;
    int max_total_time_ms;
    int progress_update_interval_ms;
    bool enable_fallback;
    int num_algorithms;
    AlgorithmConfig algorithms[8];
};

// Forward declarations
void print_uint128_decimal(uint128_t n);
uint128_t parse_decimal(const char* str);

// Helper division function for trial division
__device__ uint128_t divide_128_simple(const uint128_t& dividend, const uint128_t& divisor, uint128_t& remainder) {
    if (divisor.high == 0 && dividend.high == 0) {
        // Both fit in 64 bits
        remainder.low = dividend.low % divisor.low;
        remainder.high = 0;
        return uint128_t(dividend.low / divisor.low, 0);
    }
    
    // Simple long division algorithm
    uint128_t quotient(0, 0);
    remainder = dividend;
    
    // Find the most significant bit of divisor
    int divisor_bits = 128 - divisor.leading_zeros();
    
    for (int i = 127 - divisor_bits; i >= 0; i--) {
        // Shift remainder left by 1
        remainder = shift_left_128(remainder, 1);
        
        // Check if we can subtract divisor
        if (remainder >= divisor) {
            remainder = subtract_128(remainder, divisor);
            // Set bit in quotient
            if (i >= 64) {
                quotient.high |= (1ULL << (i - 64));
            } else {
                quotient.low |= (1ULL << i);
            }
        }
    }
    
    return quotient;
}

// Generate random 128-bit number
__device__ uint128_t generate_random_128(
    curandState_t* state,
    const uint128_t& min,
    const uint128_t& max
) {
    uint64_t low = curand(state);
    uint64_t high = curand(state);
    
    uint128_t range = subtract_128(max, min);
    uint128_t rand_val(low, high);
    
    // Simple modulo (could be optimized)
    if (range.high == 0 && range.low > 0) {
        rand_val.low %= range.low;
        rand_val.high = 0;
    }
    
    return add_128(min, rand_val);
}

// Pollard's f function
__device__ uint128_t pollards_f(
    const uint128_t& x,
    const uint128_t& c,
    const uint128_t& n,
    const Barrett128_v2& barrett
) {
    uint256_t x_squared = multiply_128_128(x, x);
    uint128_t result = barrett.reduce(x_squared);
    
    result = add_128(result, c);
    if (result >= n) {
        result = subtract_128(result, n);
    }
    
    return result;
}

// Algorithm selector
class AlgorithmSelector {
private:
    uint128_t n;
    int bit_size;
    bool is_even;
    bool is_prime_probable;
    cudaDeviceProp device_prop;
    
public:
    AlgorithmSelector(uint128_t number) : n(number) {
        cudaGetDeviceProperties(&device_prop, 0);
        analyze_number();
    }
    
    void analyze_number() {
        // Calculate bit size
        bit_size = 128 - n.leading_zeros();
        is_even = (n.low & 1) == 0;
        
        // Quick primality check (Miller-Rabin would be better)
        is_prime_probable = false; // Simplified
    }
    
    AlgorithmConfig select_algorithm() {
        AlgorithmConfig config;
        
        // Small numbers - trial division
        if (bit_size <= 20) {
            config.type = AlgorithmType::TRIAL_DIVISION;
            config.max_iterations = 1 << bit_size;
            config.num_blocks = 1;
            config.threads_per_block = 32;
        }
        // Medium numbers - Pollard's Rho Basic
        else if (bit_size <= 64) {
            config.type = AlgorithmType::POLLARDS_RHO_BASIC;
            config.max_iterations = 10000000;
            config.num_blocks = device_prop.multiProcessorCount * 2;
            config.threads_per_block = 256;
            config.pollards_rho.batch_size = 100;
        }
        // Large numbers - Pollard's Rho with Brent
        else if (bit_size <= 90) {
            config.type = AlgorithmType::POLLARDS_RHO_BRENT;
            config.max_iterations = 100000000;
            config.num_blocks = device_prop.multiProcessorCount * 4;
            config.threads_per_block = 256;
            config.pollards_rho.batch_size = 1000;
            config.pollards_rho.cycle_detection_interval = 128;
        }
        // Very large numbers - Parallel Pollard's Rho
        else {
            config.type = AlgorithmType::POLLARDS_RHO_PARALLEL;
            config.max_iterations = 1000000000;
            config.num_blocks = device_prop.multiProcessorCount * 8;
            config.threads_per_block = 256;
            config.pollards_rho.batch_size = 10000;
            config.pollards_rho.cycle_detection_interval = 256;
        }
        
        // Select reduction method
        config.use_montgomery = !is_even && (bit_size > 64);
        config.use_barrett = is_even || !config.use_montgomery;
        config.use_memory_optimization = (bit_size > 80);
        
        // Set timeouts based on complexity
        config.timeout_ms = std::min(300000, bit_size * 1000); // Max 5 minutes
        
        return config;
    }
    
    std::vector<AlgorithmConfig> get_fallback_sequence() {
        std::vector<AlgorithmConfig> sequence;
        
        // Primary algorithm
        sequence.push_back(select_algorithm());
        
        // Fallback options
        if (bit_size > 64) {
            // Try different Pollard's Rho variants
            AlgorithmConfig alt = sequence[0];
            alt.pollards_rho.batch_size *= 2;
            alt.max_iterations *= 2;
            sequence.push_back(alt);
            
            // Final fallback - basic with maximum iterations
            alt.type = AlgorithmType::POLLARDS_RHO_BASIC;
            alt.max_iterations = INT_MAX;
            sequence.push_back(alt);
        }
        
        return sequence;
    }
};

// Trial Division for small factors
__global__ void trial_division_kernel(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    int max_trial
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread tests a range of potential divisors
    int start = 2 + tid * 1000;
    int end = min(start + 1000, max_trial);
    
    for (int d = start; d < end && *factor_count < 64; d++) {
        if (d == 1) continue;
        
        uint128_t divisor(d, 0);
        uint128_t remainder;
        uint128_t quotient = divide_128_simple(n, divisor, remainder);
        
        if (remainder.is_zero()) {
            int idx = atomicAdd(factor_count, 1);
            if (idx < 64) {
                factors[idx] = divisor;
            }
        }
    }
}

// Enhanced Pollard's Rho with all optimizations
__global__ void pollards_rho_unified(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    ProgressState* progress,
    AlgorithmConfig config
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize thread state
    curandState_t rand_state;
    curand_init(clock64() + tid * 7919, tid, 0, &rand_state);
    
    // Setup Barrett reduction
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // Initialize Pollard's Rho state
    uint128_t x = generate_random_128(&rand_state, uint128_t(2, 0), n);
    uint128_t y = x;
    uint128_t c = generate_random_128(&rand_state, uint128_t(1, 0), uint128_t(1000, 0));
    
    // Brent's optimization parameters
    int m = config.pollards_rho.batch_size;
    int r = 1;
    uint128_t ys = y;
    uint128_t product(1, 0);
    
    // Main loop
    for (int i = 0; i < config.max_iterations && !progress->factor_found; i++) {
        // Update progress
        if (i % 1000 == 0) {
            update_progress_device(progress, 1000, 0, m);
        }
        
        // Pollard's Rho step
        if (config.type == AlgorithmType::POLLARDS_RHO_BRENT) {
            // Brent's variant
            if (i % r == 0) {
                ys = y;
                product = uint128_t(1, 0);
            }
            
            // Multiple steps
            for (int j = 0; j < m && i + j < r; j++) {
                y = pollards_f(y, c, n, barrett);
                uint128_t diff = (y > ys) ? subtract_128(y, ys) : subtract_128(ys, y);
                
                uint256_t prod = multiply_128_128(product, diff);
                product = barrett.reduce(prod);
            }
            
            // Check GCD
            uint128_t g = gcd_128(product, n);
            if (g > uint128_t(1, 0) && g < n) {
                int idx = atomicAdd(factor_count, 1);
                if (idx < 64) {
                    factors[idx] = g;
                    progress->factor_found = true;
                }
                break;
            }
            
            if (i % (r * 2) == 0) {
                r *= 2;
            }
        } else {
            // Basic variant
            x = pollards_f(x, c, n, barrett);
            y = pollards_f(pollards_f(y, c, n, barrett), c, n, barrett);
            
            uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
            uint128_t g = gcd_128(diff, n);
            
            if (g > uint128_t(1, 0) && g < n) {
                int idx = atomicAdd(factor_count, 1);
                if (idx < 64) {
                    factors[idx] = g;
                    progress->factor_found = true;
                }
                break;
            }
        }
        
        // Adaptive parameter change
        if (i % (10000 * (tid % 10 + 1)) == 0) {
            c = generate_random_128(&rand_state, uint128_t(1, 0), uint128_t(10000, 0));
            x = generate_random_128(&rand_state, uint128_t(2, 0), n);
            y = x;
        }
    }
}

// Main unified factorizer class
class UnifiedFactorizer {
private:
    FactorizerConfig config;
    FactorizationResult result;
    std::atomic<AlgorithmState> current_state;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    ProgressReporter* progress_reporter;
    
    // GPU resources
    uint128_t* d_factors;
    int* d_factor_count;
    ProgressState* d_progress;
    
public:
    UnifiedFactorizer(const FactorizerConfig& cfg) 
        : config(cfg), current_state(AlgorithmState::NOT_STARTED) {
        
        // Initialize result
        memset(&result, 0, sizeof(result));
        
        // Allocate GPU resources
        cudaMalloc(&d_factors, 64 * sizeof(uint128_t));
        cudaMalloc(&d_factor_count, sizeof(int));
        cudaMemset(d_factor_count, 0, sizeof(int));
        
        progress_reporter = nullptr;
    }
    
    ~UnifiedFactorizer() {
        if (d_factors) cudaFree(d_factors);
        if (d_factor_count) cudaFree(d_factor_count);
        if (progress_reporter) delete progress_reporter;
    }
    
    FactorizationResult factorize(uint128_t n) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (config.verbose) {
            print_header(n);
        }
        
        // Initialize progress reporter
        if (config.show_progress) {
            int total_threads = config.algorithms[0].num_blocks * 
                               config.algorithms[0].threads_per_block;
            progress_reporter = new ProgressReporter(n, total_threads, config.verbose);
            d_progress = progress_reporter->get_device_pointer();
        }
        
        // Algorithm selection
        AlgorithmSelector selector(n);
        std::vector<AlgorithmConfig> algorithm_sequence;
        
        if (config.auto_algorithm) {
            algorithm_sequence = selector.get_fallback_sequence();
        } else {
            for (int i = 0; i < config.num_algorithms; i++) {
                algorithm_sequence.push_back(config.algorithms[i]);
            }
        }
        
        // Execute algorithms in sequence
        bool factor_found = false;
        for (size_t i = 0; i < algorithm_sequence.size() && !factor_found; i++) {
            if (config.verbose) {
                printf("\nTrying algorithm: %s\n", 
                       get_algorithm_name(algorithm_sequence[i].type));
            }
            
            current_state = AlgorithmState::RUNNING;
            result.algorithm_used = algorithm_sequence[i].type;
            result.algorithm_switches = i;
            
            // Run algorithm
            factor_found = run_algorithm(n, algorithm_sequence[i]);
            
            if (factor_found) {
                current_state = AlgorithmState::COMPLETED;
            } else if (i < algorithm_sequence.size() - 1) {
                current_state = AlgorithmState::TRANSITIONING;
                if (config.verbose) {
                    printf("Algorithm did not find factors, transitioning...\n");
                }
            } else {
                current_state = AlgorithmState::FAILED;
            }
        }
        
        // Calculate final metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        result.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        // Copy factors from device
        cudaMemcpy(&result.factor_count, d_factor_count, sizeof(int), 
                   cudaMemcpyDeviceToHost);
        if (result.factor_count > 0) {
            cudaMemcpy(result.factors, d_factors, 
                       result.factor_count * sizeof(uint128_t), 
                       cudaMemcpyDeviceToHost);
            result.is_complete = verify_factorization(n);
        }
        
        if (config.verbose) {
            print_results(n);
        }
        
        return result;
    }
    
private:
    bool run_algorithm(uint128_t n, const AlgorithmConfig& algo_config) {
        bool success = false;
        
        switch (algo_config.type) {
            case AlgorithmType::TRIAL_DIVISION:
                success = run_trial_division(n, algo_config);
                break;
                
            case AlgorithmType::POLLARDS_RHO_BASIC:
            case AlgorithmType::POLLARDS_RHO_BRENT:
            case AlgorithmType::POLLARDS_RHO_PARALLEL:
                success = run_pollards_rho(n, algo_config);
                break;
                
            default:
                if (config.verbose) {
                    printf("Algorithm not implemented yet\n");
                }
                break;
        }
        
        return success;
    }
    
    bool run_trial_division(uint128_t n, const AlgorithmConfig& algo_config) {
        int max_trial = std::min(1000000, algo_config.max_iterations);
        
        trial_division_kernel<<<algo_config.num_blocks, algo_config.threads_per_block>>>(
            n, d_factors, d_factor_count, max_trial
        );
        
        cudaDeviceSynchronize();
        
        int factor_count;
        cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        return factor_count > 0;
    }
    
    bool run_pollards_rho(uint128_t n, const AlgorithmConfig& algo_config) {
        // Reset factor count
        cudaMemset(d_factor_count, 0, sizeof(int));
        
        // Launch kernel
        pollards_rho_unified<<<algo_config.num_blocks, algo_config.threads_per_block>>>(
            n, d_factors, d_factor_count, d_progress, algo_config
        );
        
        // Monitor progress
        auto start = std::chrono::steady_clock::now();
        bool timeout = false;
        int factor_count = 0;
        
        while (factor_count == 0 && !timeout) {
            if (progress_reporter) {
                progress_reporter->update_and_report();
            }
            
            // Check if kernel finished
            if (cudaStreamQuery(0) == cudaSuccess) {
                break;
            }
            
            // Check timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start).count();
            if (elapsed > algo_config.timeout_ms) {
                timeout = true;
                if (config.verbose) {
                    printf("Algorithm timeout after %ld ms\n", elapsed);
                }
            }
            
            // Check factor count
            cudaMemcpy(&factor_count, d_factor_count, sizeof(int), 
                       cudaMemcpyDeviceToHost);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        cudaDeviceSynchronize();
        
        // Final check
        cudaMemcpy(&factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Get total iterations from progress state
        if (progress_reporter) {
            ProgressState h_progress;
            cudaMemcpy(&h_progress, d_progress, sizeof(ProgressState), cudaMemcpyDeviceToHost);
            result.total_iterations = h_progress.total_iterations;
        }
        
        return factor_count > 0;
    }
    
    bool verify_factorization(uint128_t n) {
        if (result.factor_count == 0) return false;
        
        uint128_t product(1, 0);
        for (int i = 0; i < result.factor_count; i++) {
            uint256_t temp = multiply_128_128(product, result.factors[i]);
            product = uint128_t(temp.word[0], temp.word[1]);
        }
        
        return product == n;
    }
    
    void print_header(uint128_t n) {
        printf("\n");
        printf("==================================================\n");
        printf("    CUDA Factorizer v%s - Unified Edition\n", VERSION_STRING);
        printf("==================================================\n");
        printf("Target number: ");
        print_uint128_decimal(n);
        printf("\n");
        printf("Bit size: %d\n", 128 - n.leading_zeros());
        printf("--------------------------------------------------\n\n");
    }
    
    void print_results(uint128_t n) {
        printf("\n--------------------------------------------------\n");
        printf("                 RESULTS SUMMARY\n");
        printf("--------------------------------------------------\n");
        
        if (result.factor_count > 0) {
            printf("✓ Factorization successful!\n\n");
            printf("Factors found: %d\n", result.factor_count);
            
            for (int i = 0; i < result.factor_count; i++) {
                printf("  Factor %d: ", i + 1);
                print_uint128_decimal(result.factors[i]);
                printf("\n");
            }
            
            if (result.is_complete) {
                printf("\n✓ Factorization verified: product equals input\n");
            } else {
                printf("\n✗ Warning: factorization incomplete\n");
            }
        } else {
            printf("✗ No factors found\n");
            if (strlen(result.error_message) > 0) {
                printf("Error: %s\n", result.error_message);
            }
        }
        
        printf("\nPerformance metrics:\n");
        printf("  Algorithm used: %s\n", get_algorithm_name(result.algorithm_used));
        printf("  Algorithm switches: %d\n", result.algorithm_switches);
        printf("  Total time: %.3f seconds\n", result.total_time_ms / 1000.0);
        printf("  Total iterations: %llu\n", (unsigned long long)result.total_iterations);
        if (result.total_time_ms > 0) {
            printf("  Iterations/second: %.2f million\n", 
                   (result.total_iterations / result.total_time_ms) / 1000.0);
        }
        printf("  GPU threads used: %d\n", result.gpu_threads_used);
        printf("==================================================\n\n");
    }
    
    const char* get_algorithm_name(AlgorithmType type) {
        switch (type) {
            case AlgorithmType::TRIAL_DIVISION: return "Trial Division";
            case AlgorithmType::POLLARDS_RHO_BASIC: return "Pollard's Rho (Basic)";
            case AlgorithmType::POLLARDS_RHO_BRENT: return "Pollard's Rho (Brent)";
            case AlgorithmType::POLLARDS_RHO_PARALLEL: return "Pollard's Rho (Parallel)";
            case AlgorithmType::QUADRATIC_SIEVE: return "Quadratic Sieve";
            case AlgorithmType::ELLIPTIC_CURVE: return "Elliptic Curve Method";
            default: return "Unknown";
        }
    }
};

// Parse decimal string to uint128
uint128_t parse_decimal(const char* str) {
    uint128_t result(0, 0);
    uint128_t ten(10, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            uint256_t temp = multiply_128_128(result, ten);
            result = uint128_t(temp.word[0], temp.word[1]);
            result = add_128(result, uint128_t(str[i] - '0', 0));
        }
    }
    
    return result;
}

// Print uint128 in decimal
void print_uint128_decimal(uint128_t n) {
    if (n.high == 0) {
        printf("%llu", n.low);
        return;
    }
    
    char buffer[40];
    int pos = 39;
    buffer[pos] = '\0';
    
    uint128_t ten(10, 0);
    while (!n.is_zero() && pos > 0) {
        uint128_t quotient(0, 0);
        uint64_t remainder = 0;
        
        if (n.high > 0) {
            remainder = n.high % 10;
            quotient.high = n.high / 10;
        }
        
        uint64_t temp = remainder * (1ULL << 32) * (1ULL << 32) + n.low;
        quotient.low = temp / 10;
        remainder = temp % 10;
        
        buffer[--pos] = '0' + remainder;
        n = quotient;
    }
    
    printf("%s", &buffer[pos]);
}

// Get default configuration
FactorizerConfig get_default_config() {
    FactorizerConfig config;
    config.verbose = true;
    config.show_progress = true;
    config.auto_algorithm = true;
    config.gpu_monitoring = true;
    config.max_total_time_ms = 300000; // 5 minutes
    config.progress_update_interval_ms = 1000;
    config.enable_fallback = true;
    config.num_algorithms = 0; // Auto mode
    return config;
}

// Print usage
void print_usage(const char* program_name) {
    printf("Usage: %s <number> [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  -q, --quiet         Suppress verbose output\n");
    printf("  -np, --no-progress  Disable progress reporting\n");
    printf("  -a, --algorithm     Force specific algorithm\n");
    printf("  -t, --timeout       Set timeout in seconds\n");
    printf("  -b, --blocks        Number of CUDA blocks\n");
    printf("  -th, --threads      Threads per block\n");
    printf("  -h, --help          Show this help\n");
    printf("\nExample:\n");
    printf("  %s 15482526220500967432610341\n", program_name);
    printf("  %s 123456789 -a pollards_rho_brent -t 60\n", program_name);
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Parse number
    uint128_t n;
    if (strcmp(argv[1], "test") == 0) {
        // Use the 26-digit test case
        n = parse_decimal("15482526220500967432610341");
    } else {
        n = parse_decimal(argv[1]);
    }
    
    // Get configuration
    FactorizerConfig config = get_default_config();
    
    // Parse command line options
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            config.verbose = false;
        } else if (strcmp(argv[i], "-np") == 0 || strcmp(argv[i], "--no-progress") == 0) {
            config.show_progress = false;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) {
            if (i + 1 < argc) {
                config.max_total_time_ms = atoi(argv[++i]) * 1000;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("Error: No CUDA-capable devices found\n");
        return 1;
    }
    
    // Set device
    cudaSetDevice(0);
    
    // Create and run factorizer
    UnifiedFactorizer factorizer(config);
    FactorizationResult result = factorizer.factorize(n);
    
    // Return success/failure
    return result.is_complete ? 0 : 1;
}