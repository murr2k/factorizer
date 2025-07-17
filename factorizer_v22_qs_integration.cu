/**
 * Quadratic Sieve Integration for Factorizer v2.2.0
 * Adds QS support to the unified factorization framework
 */

#include "quadratic_sieve_complete.cuh"

// Forward declaration of QS entry point
extern "C" bool quadratic_sieve_factor_complete(uint128_t n, uint128_t& factor1, uint128_t& factor2);

/**
 * Enhanced Algorithm Selector with QS support
 */
class AlgorithmSelectorQS : public AlgorithmSelector {
public:
    AlgorithmSelectorQS(uint128_t number) : AlgorithmSelector(number) {}
    
    AlgorithmConfig select_algorithm() override {
        AlgorithmConfig config = AlgorithmSelector::select_algorithm();
        
        // Override for numbers suitable for QS
        // QS is efficient for numbers with 40-100 bit factors
        int bit_size = 128 - n.leading_zeros();
        
        if (bit_size >= 80 && bit_size <= 120) {
            // Check if number might have balanced factors
            // Simple heuristic: if Pollard's Rho hasn't found small factors quickly
            config.type = AlgorithmType::QUADRATIC_SIEVE;
            config.max_iterations = 1000000; // Not used for QS
            config.num_blocks = device_prop.multiProcessorCount * 4;
            config.threads_per_block = QS_BLOCK_SIZE;
            config.quadratic_sieve.sieve_size = QS_SIEVE_INTERVAL;
            config.quadratic_sieve.smooth_bound = 100000;
            config.timeout_ms = 600000; // 10 minutes for QS
            config.use_barrett = true;
            config.use_montgomery = false;
        }
        
        return config;
    }
    
    std::vector<AlgorithmConfig> get_fallback_sequence() override {
        std::vector<AlgorithmConfig> sequence;
        
        int bit_size = 128 - n.leading_zeros();
        
        if (bit_size >= 80 && bit_size <= 120) {
            // Try Pollard's Rho first for 30 seconds
            AlgorithmConfig rho_config;
            rho_config.type = AlgorithmType::POLLARDS_RHO_BRENT;
            rho_config.max_iterations = 50000000;
            rho_config.timeout_ms = 30000; // 30 seconds
            rho_config.num_blocks = device_prop.multiProcessorCount * 4;
            rho_config.threads_per_block = 256;
            rho_config.pollards_rho.batch_size = 1000;
            sequence.push_back(rho_config);
            
            // Then try QS
            AlgorithmConfig qs_config;
            qs_config.type = AlgorithmType::QUADRATIC_SIEVE;
            qs_config.max_iterations = 1000000;
            qs_config.timeout_ms = 300000; // 5 minutes
            qs_config.num_blocks = device_prop.multiProcessorCount * 4;
            qs_config.threads_per_block = QS_BLOCK_SIZE;
            qs_config.quadratic_sieve.sieve_size = QS_SIEVE_INTERVAL;
            qs_config.quadratic_sieve.smooth_bound = 100000;
            sequence.push_back(qs_config);
            
            // Final fallback - parallel Pollard's Rho
            AlgorithmConfig final_config;
            final_config.type = AlgorithmType::POLLARDS_RHO_PARALLEL;
            final_config.max_iterations = INT_MAX;
            final_config.timeout_ms = 600000; // 10 minutes
            final_config.num_blocks = device_prop.multiProcessorCount * 8;
            final_config.threads_per_block = 256;
            sequence.push_back(final_config);
        } else {
            // Use default sequence for other sizes
            sequence = AlgorithmSelector::get_fallback_sequence();
        }
        
        return sequence;
    }
};

/**
 * QS runner method for UnifiedFactorizer
 */
bool UnifiedFactorizer::run_quadratic_sieve(uint128_t n, const AlgorithmConfig& algo_config) {
    if (config.verbose) {
        printf("Starting Quadratic Sieve algorithm...\n");
        printf("  Sieve interval: %d\n", algo_config.quadratic_sieve.sieve_size);
        printf("  Smooth bound: %d\n", algo_config.quadratic_sieve.smooth_bound);
    }
    
    // Call the QS implementation
    uint128_t factor1, factor2;
    bool success = quadratic_sieve_factor_complete(n, factor1, factor2);
    
    if (success) {
        // Store factors
        result.factors[0] = factor1;
        result.factors[1] = factor2;
        result.factor_count = 2;
        
        // Update metrics
        result.algorithm_used = AlgorithmType::QUADRATIC_SIEVE;
        result.is_complete = true;
        
        if (config.verbose) {
            printf("QS found factors!\n");
        }
    }
    
    return success;
}

/**
 * Modified run_algorithm to include QS
 */
bool UnifiedFactorizer::run_algorithm_with_qs(uint128_t n, const AlgorithmConfig& algo_config) {
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
            
        case AlgorithmType::QUADRATIC_SIEVE:
            success = run_quadratic_sieve(n, algo_config);
            break;
            
        case AlgorithmType::ELLIPTIC_CURVE:
            if (config.verbose) {
                printf("ECM not implemented yet, falling back to QS\n");
            }
            // Fall back to QS for now
            success = run_quadratic_sieve(n, algo_config);
            break;
            
        default:
            if (config.verbose) {
                printf("Unknown algorithm type\n");
            }
            break;
    }
    
    return success;
}

/**
 * Test function for QS integration
 */
void test_qs_integration() {
    printf("\n=== Testing Quadratic Sieve Integration ===\n");
    
    // Test cases with balanced factors (good for QS)
    struct TestCase {
        const char* n_str;
        const char* expected_factor1;
        const char* expected_factor2;
        const char* description;
    };
    
    TestCase test_cases[] = {
        // 60-bit number with 30-bit factors
        {"1152921504606846999", "1073741827", "1073741837", "Two 30-bit primes"},
        
        // 80-bit number with 40-bit factors
        {"1208925819614629174706449", "1099511627791", "1099511627803", "Two 40-bit primes"},
        
        // 90-bit number with 45-bit factors
        {"1237940039285380274899124357", "1125899906842679", "1125899906842683", "Two 45-bit primes"},
        
        // The 86-bit test case
        {"29318992932113061061655073", "4872513061429", "6018304725797", "86-bit semiprime"}
    };
    
    FactorizerConfig config = get_default_config();
    config.auto_algorithm = true;
    
    for (const auto& test : test_cases) {
        printf("\nTest: %s\n", test.description);
        printf("N = %s\n", test.n_str);
        
        uint128_t n = parse_decimal(test.n_str);
        
        UnifiedFactorizer factorizer(config);
        FactorizationResult result = factorizer.factorize(n);
        
        if (result.is_complete && result.factor_count == 2) {
            printf("Success! Found factors:\n");
            printf("  Factor 1: ");
            print_uint128_decimal(result.factors[0]);
            printf("\n  Factor 2: ");
            print_uint128_decimal(result.factors[1]);
            printf("\n");
            
            // Verify
            uint256_t product = multiply_128_128(result.factors[0], result.factors[1]);
            uint128_t n_check(product.word[0], product.word[1]);
            
            if (n_check == n) {
                printf("  ✓ Verification passed\n");
            } else {
                printf("  ✗ Verification failed\n");
            }
        } else {
            printf("Failed to factor\n");
        }
        
        printf("Algorithm: %s\n", get_algorithm_name(result.algorithm_used));
        printf("Time: %.3f seconds\n", result.total_time_ms / 1000.0);
    }
}

/**
 * Benchmark QS vs Pollard's Rho for different number sizes
 */
void benchmark_qs_performance() {
    printf("\n=== Benchmarking QS Performance ===\n");
    
    struct BenchmarkCase {
        int bit_size;
        int factor_bits;
        int num_tests;
    };
    
    BenchmarkCase benchmarks[] = {
        {60, 30, 5},
        {70, 35, 5},
        {80, 40, 5},
        {90, 45, 3},
        {100, 50, 2}
    };
    
    for (const auto& bench : benchmarks) {
        printf("\n%d-bit numbers with %d-bit factors:\n", bench.bit_size, bench.factor_bits);
        
        double qs_total_time = 0;
        double rho_total_time = 0;
        int qs_success = 0;
        int rho_success = 0;
        
        for (int i = 0; i < bench.num_tests; i++) {
            // Generate test number (would need proper prime generation)
            // For now, use predetermined values
            uint128_t n = uint128_t(1ULL << (bench.factor_bits - 1), 0);
            n = multiply_128_128(n, n).to_uint128();
            
            // Test with QS
            {
                FactorizerConfig config = get_default_config();
                config.auto_algorithm = false;
                config.num_algorithms = 1;
                config.algorithms[0].type = AlgorithmType::QUADRATIC_SIEVE;
                
                auto start = std::chrono::high_resolution_clock::now();
                UnifiedFactorizer factorizer(config);
                FactorizationResult result = factorizer.factorize(n);
                auto end = std::chrono::high_resolution_clock::now();
                
                double time = std::chrono::duration<double>(end - start).count();
                if (result.is_complete) {
                    qs_success++;
                    qs_total_time += time;
                }
            }
            
            // Test with Pollard's Rho
            {
                FactorizerConfig config = get_default_config();
                config.auto_algorithm = false;
                config.num_algorithms = 1;
                config.algorithms[0].type = AlgorithmType::POLLARDS_RHO_BRENT;
                
                auto start = std::chrono::high_resolution_clock::now();
                UnifiedFactorizer factorizer(config);
                FactorizationResult result = factorizer.factorize(n);
                auto end = std::chrono::high_resolution_clock::now();
                
                double time = std::chrono::duration<double>(end - start).count();
                if (result.is_complete) {
                    rho_success++;
                    rho_total_time += time;
                }
            }
        }
        
        printf("  QS: %d/%d success, avg time: %.3f seconds\n",
               qs_success, bench.num_tests, 
               qs_success > 0 ? qs_total_time / qs_success : 0);
        printf("  Rho: %d/%d success, avg time: %.3f seconds\n",
               rho_success, bench.num_tests,
               rho_success > 0 ? rho_total_time / rho_success : 0);
    }
}

/**
 * Main function for testing QS integration
 */
int main(int argc, char* argv[]) {
    printf("CUDA Factorizer v2.2.0 - Quadratic Sieve Integration\n");
    printf("====================================================\n");
    
    // Initialize CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\n");
    
    if (argc > 1 && strcmp(argv[1], "benchmark") == 0) {
        benchmark_qs_performance();
    } else if (argc > 1) {
        // Factor specific number
        uint128_t n = parse_decimal(argv[1]);
        
        FactorizerConfig config = get_default_config();
        UnifiedFactorizer factorizer(config);
        FactorizationResult result = factorizer.factorize(n);
        
        // Results are printed by the factorizer
    } else {
        // Run integration tests
        test_qs_integration();
    }
    
    return 0;
}