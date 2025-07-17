/**
 * Factorizer v2.2.0 - Intelligent Algorithm Selector Implementation
 * 
 * This module implements the algorithm selection logic based on
 * number characteristics and available resources.
 */

#ifndef FACTORIZER_V22_SELECTOR_CUH
#define FACTORIZER_V22_SELECTOR_CUH

#include "factorizer_v22_architecture.h"
#include <cmath>
#include <algorithm>

//=============================================================================
// Algorithm Characteristics Database
//=============================================================================

static const algorithm_info_t algorithm_database[ALGO_COUNT] = {
    [ALGO_AUTO] = {
        .id = ALGO_AUTO,
        .name = "Automatic Selection",
        .min_digit_efficient = 0,
        .max_digit_efficient = 999,
        .gpu_memory_mb = 0,
        .complexity_factor = 0.0f,
        .supports_partial = true,
        .requires_smoothness = false
    },
    
    [ALGO_TRIAL_DIVISION] = {
        .id = ALGO_TRIAL_DIVISION,
        .name = "Trial Division",
        .min_digit_efficient = 1,
        .max_digit_efficient = 8,
        .gpu_memory_mb = 128,
        .complexity_factor = 1.0f,
        .supports_partial = true,
        .requires_smoothness = false
    },
    
    [ALGO_POLLARD_RHO] = {
        .id = ALGO_POLLARD_RHO,
        .name = "Pollard's Rho",
        .min_digit_efficient = 8,
        .max_digit_efficient = 20,
        .gpu_memory_mb = 256,
        .complexity_factor = 2.5f,
        .supports_partial = false,
        .requires_smoothness = false
    },
    
    [ALGO_POLLARD_P1] = {
        .id = ALGO_POLLARD_P1,
        .name = "Pollard's p-1",
        .min_digit_efficient = 10,
        .max_digit_efficient = 25,
        .gpu_memory_mb = 512,
        .complexity_factor = 3.0f,
        .supports_partial = false,
        .requires_smoothness = true
    },
    
    [ALGO_ECM] = {
        .id = ALGO_ECM,
        .name = "Elliptic Curve Method",
        .min_digit_efficient = 15,
        .max_digit_efficient = 30,
        .gpu_memory_mb = 1024,
        .complexity_factor = 4.5f,
        .supports_partial = true,
        .requires_smoothness = false
    },
    
    [ALGO_QUADRATIC_SIEVE] = {
        .id = ALGO_QUADRATIC_SIEVE,
        .name = "Quadratic Sieve",
        .min_digit_efficient = 20,
        .max_digit_efficient = 40,
        .gpu_memory_mb = 2048,
        .complexity_factor = 6.0f,
        .supports_partial = false,
        .requires_smoothness = true
    },
    
    [ALGO_GNFS] = {
        .id = ALGO_GNFS,
        .name = "General Number Field Sieve",
        .min_digit_efficient = 40,
        .max_digit_efficient = 100,
        .gpu_memory_mb = 8192,
        .complexity_factor = 10.0f,
        .supports_partial = false,
        .requires_smoothness = true
    },
    
    [ALGO_HYBRID] = {
        .id = ALGO_HYBRID,
        .name = "Hybrid Multi-Algorithm",
        .min_digit_efficient = 15,
        .max_digit_efficient = 50,
        .gpu_memory_mb = 4096,
        .complexity_factor = 5.0f,
        .supports_partial = true,
        .requires_smoothness = false
    }
};

//=============================================================================
// Number Analysis Functions
//=============================================================================

/**
 * Count the number of decimal digits in a 128-bit number
 */
__host__ __device__ int count_digits(const factor_t* n) {
    if (n->is_zero()) return 1;
    
    // Approximate using bit count
    int bits = 128 - n->leading_zeros();
    // log10(2^bits) ≈ bits * 0.30103
    int digits = (int)(bits * 0.30103) + 1;
    
    // Refine if necessary (host only)
    #ifndef __CUDA_ARCH__
    factor_t ten(10);
    factor_t power(1);
    int actual_digits = 1;
    
    while (power <= *n && actual_digits < 40) {
        power = multiply_128_128(power, ten).low_128();
        actual_digits++;
    }
    return actual_digits - 1;
    #else
    return digits;
    #endif
}

/**
 * Estimate smoothness of a number
 * Returns a value between 0.0 (not smooth) and 1.0 (very smooth)
 */
float estimate_smoothness(const factor_t* n, int small_factor_bound) {
    // Simple heuristic based on small factor removal
    factor_t remaining = *n;
    int factors_removed = 0;
    
    // Check powers of 2
    while ((remaining.low & 1) == 0) {
        remaining = shift_right_128(remaining, 1);
        factors_removed++;
    }
    
    // If we removed many small factors, number might be smooth
    float smoothness = 1.0f - (float)count_digits(&remaining) / count_digits(n);
    return fminf(fmaxf(smoothness, 0.0f), 1.0f);
}

/**
 * Check if number has special form
 */
bool has_special_form(const factor_t* n) {
    // Check for Mersenne numbers (2^p - 1)
    factor_t n_plus_1 = add_128(*n, factor_t(1));
    int trailing_zeros = 0;
    factor_t temp = n_plus_1;
    
    while ((temp.low & 1) == 0 && !temp.is_zero()) {
        temp = shift_right_128(temp, 1);
        trailing_zeros++;
    }
    
    if (temp == factor_t(1)) {
        return true; // Mersenne number
    }
    
    // Check for Fermat numbers (2^(2^n) + 1)
    factor_t n_minus_1 = subtract_128(*n, factor_t(1));
    temp = n_minus_1;
    trailing_zeros = 0;
    
    while ((temp.low & 1) == 0 && !temp.is_zero()) {
        temp = shift_right_128(temp, 1);
        trailing_zeros++;
    }
    
    // Check if trailing_zeros is a power of 2
    if (temp == factor_t(1) && (trailing_zeros & (trailing_zeros - 1)) == 0) {
        return true; // Fermat number
    }
    
    return false;
}

//=============================================================================
// Algorithm Suitability Scoring
//=============================================================================

/**
 * Calculate suitability score for Pollard's Rho
 */
float pollard_rho_suitability(const factor_t* number, const number_characteristics_t* chars) {
    float score = 0.0f;
    
    // Optimal for 8-20 digit numbers
    if (chars->digit_count >= 8 && chars->digit_count <= 20) {
        score = 1.0f - fabsf(chars->digit_count - 14.0f) / 6.0f;
    } else if (chars->digit_count < 8) {
        score = 0.3f; // Can work but not optimal
    } else if (chars->digit_count <= 25) {
        score = 0.6f - (chars->digit_count - 20) * 0.05f;
    }
    
    // Penalty for special forms (better algorithms exist)
    if (chars->has_special_form) {
        score *= 0.7f;
    }
    
    // Bonus for non-smooth numbers
    score *= (1.0f + (1.0f - chars->smoothness_estimate) * 0.3f);
    
    return fmaxf(score, 0.0f);
}

/**
 * Calculate suitability score for ECM
 */
float ecm_suitability(const factor_t* number, const number_characteristics_t* chars) {
    float score = 0.0f;
    
    // Optimal for finding 15-30 digit factors
    if (chars->digit_count >= 15 && chars->digit_count <= 30) {
        score = 1.0f - fabsf(chars->digit_count - 22.5f) / 7.5f;
    } else if (chars->digit_count > 30 && chars->digit_count <= 40) {
        score = 0.5f;
    }
    
    // ECM is good at finding medium-sized factors
    // Assume factors are roughly sqrt(n) in size
    int expected_factor_digits = (chars->digit_count + 1) / 2;
    if (expected_factor_digits >= 12 && expected_factor_digits <= 25) {
        score *= 1.2f;
    }
    
    return fminf(fmaxf(score, 0.0f), 1.0f);
}

/**
 * Calculate suitability score for Quadratic Sieve
 */
float quadratic_sieve_suitability(const factor_t* number, const number_characteristics_t* chars) {
    float score = 0.0f;
    
    // Optimal for 20-40 digit numbers
    if (chars->digit_count >= 20 && chars->digit_count <= 40) {
        score = 1.0f - fabsf(chars->digit_count - 30.0f) / 10.0f;
    } else if (chars->digit_count > 40 && chars->digit_count <= 50) {
        score = 0.4f;
    }
    
    // QS benefits from smooth numbers
    score *= (1.0f + chars->smoothness_estimate * 0.5f);
    
    // Penalty for probable primes (QS assumes composite)
    if (chars->is_probable_prime) {
        score *= 0.1f;
    }
    
    return fmaxf(score, 0.0f);
}

//=============================================================================
// Algorithm Selection Implementation
//=============================================================================

class IntelligentAlgorithmSelector {
private:
    // Historical performance data (could be persisted)
    struct PerformanceRecord {
        int digit_count;
        algorithm_id_t algorithm;
        double success_rate;
        double avg_time;
        int sample_count;
    };
    
    PerformanceRecord history[ALGO_COUNT][50]; // Up to 50 digit ranges
    
public:
    /**
     * Select the best algorithm for a given number
     */
    algorithm_id_t select_algorithm(
        const factor_t* number,
        const number_characteristics_t* chars,
        const selection_criteria_t* criteria
    ) {
        // Handle explicit preference
        if (criteria && criteria->preferred != ALGO_AUTO) {
            return criteria->preferred;
        }
        
        // Score each algorithm
        struct ScoredAlgorithm {
            algorithm_id_t id;
            float score;
        } scores[ALGO_COUNT - 1]; // Exclude AUTO
        
        int valid_count = 0;
        
        for (int i = 1; i < ALGO_COUNT; i++) {
            algorithm_id_t algo = (algorithm_id_t)i;
            
            // Skip excluded algorithms
            if (criteria) {
                bool excluded = false;
                for (int j = 0; j < ALGO_COUNT && criteria->exclude[j] != ALGO_AUTO; j++) {
                    if (criteria->exclude[j] == algo) {
                        excluded = true;
                        break;
                    }
                }
                if (excluded) continue;
            }
            
            // Check memory constraint
            if (criteria && criteria->memory_limit_mb > 0 &&
                algorithm_database[algo].gpu_memory_mb > criteria->memory_limit_mb) {
                continue;
            }
            
            // Calculate base suitability score
            float score = 0.0f;
            switch (algo) {
                case ALGO_POLLARD_RHO:
                    score = pollard_rho_suitability(number, chars);
                    break;
                case ALGO_ECM:
                    score = ecm_suitability(number, chars);
                    break;
                case ALGO_QUADRATIC_SIEVE:
                    score = quadratic_sieve_suitability(number, chars);
                    break;
                case ALGO_TRIAL_DIVISION:
                    score = (chars->digit_count <= 8) ? 1.0f : 0.1f;
                    break;
                case ALGO_POLLARD_P1:
                    score = chars->smoothness_estimate * 0.8f;
                    break;
                default:
                    score = 0.5f; // Default score for unimplemented
            }
            
            // Adjust for speed preference
            if (criteria && criteria->prefer_speed) {
                score *= (2.0f - algorithm_database[algo].complexity_factor / 10.0f);
            }
            
            // Store scored algorithm
            if (score > 0.0f) {
                scores[valid_count].id = algo;
                scores[valid_count].score = score;
                valid_count++;
            }
        }
        
        // Sort by score
        std::sort(scores, scores + valid_count,
            [](const ScoredAlgorithm& a, const ScoredAlgorithm& b) {
                return a.score > b.score;
            });
        
        // Return best algorithm or default to Pollard's Rho
        return (valid_count > 0) ? scores[0].id : ALGO_POLLARD_RHO;
    }
    
    /**
     * Get ranked list of algorithms
     */
    void rank_algorithms(
        const factor_t* number,
        const number_characteristics_t* chars,
        algorithm_id_t* ranking,
        float* scores_out,
        int count
    ) {
        selection_criteria_t default_criteria = {0};
        
        // Get all algorithm scores
        struct ScoredAlgorithm {
            algorithm_id_t id;
            float score;
        } scores[ALGO_COUNT - 1];
        
        int valid_count = 0;
        for (int i = 1; i < ALGO_COUNT; i++) {
            algorithm_id_t algo = (algorithm_id_t)i;
            float score = 0.0f;
            
            switch (algo) {
                case ALGO_POLLARD_RHO:
                    score = pollard_rho_suitability(number, chars);
                    break;
                case ALGO_ECM:
                    score = ecm_suitability(number, chars);
                    break;
                case ALGO_QUADRATIC_SIEVE:
                    score = quadratic_sieve_suitability(number, chars);
                    break;
                case ALGO_TRIAL_DIVISION:
                    score = (chars->digit_count <= 8) ? 1.0f : 0.1f;
                    break;
                default:
                    score = 0.3f;
            }
            
            scores[valid_count].id = algo;
            scores[valid_count].score = score;
            valid_count++;
        }
        
        // Sort by score
        std::sort(scores, scores + valid_count,
            [](const ScoredAlgorithm& a, const ScoredAlgorithm& b) {
                return a.score > b.score;
            });
        
        // Fill output arrays
        int output_count = std::min(count, valid_count);
        for (int i = 0; i < output_count; i++) {
            ranking[i] = scores[i].id;
            if (scores_out) {
                scores_out[i] = scores[i].score;
            }
        }
    }
    
    /**
     * Update performance model with results
     */
    void update_model(
        const factor_t* number,
        algorithm_id_t algorithm,
        const algorithm_result_t* result
    ) {
        // Update historical performance data
        int digit_count = count_digits(number);
        int digit_bucket = std::min(digit_count, 49);
        
        PerformanceRecord* record = &history[algorithm][digit_bucket];
        
        // Update success rate
        float success = result->factor_count > 0 ? 1.0f : 0.0f;
        if (record->sample_count == 0) {
            record->digit_count = digit_count;
            record->algorithm = algorithm;
            record->success_rate = success;
            record->avg_time = result->runtime_seconds;
            record->sample_count = 1;
        } else {
            // Exponential moving average
            float alpha = 0.1f;
            record->success_rate = record->success_rate * (1 - alpha) + success * alpha;
            record->avg_time = record->avg_time * (1 - alpha) + result->runtime_seconds * alpha;
            record->sample_count++;
        }
    }
};

//=============================================================================
// Factor Size Estimation
//=============================================================================

/**
 * Estimate the size of smallest remaining factor
 */
factor_size_estimate_t estimate_factor_size(
    const factor_t* number,
    const factor_t* known_factors,
    int known_count
) {
    factor_size_estimate_t estimate;
    
    // Calculate remaining number after dividing out known factors
    factor_t remaining = *number;
    for (int i = 0; i < known_count; i++) {
        // Simple division - would need proper implementation
        // This is a placeholder
    }
    
    int remaining_digits = count_digits(&remaining);
    
    // Heuristic: smallest factor is likely around sqrt(remaining)
    estimate.min_digits = (remaining_digits + 1) / 2 - 2;
    estimate.max_digits = (remaining_digits + 1) / 2 + 2;
    estimate.confidence = 0.7f;
    
    // Special cases
    if (remaining_digits <= 10) {
        estimate.confidence = 0.9f;
        estimate.reasoning = "Small number - factors well bounded";
    } else if (remaining_digits <= 20) {
        estimate.confidence = 0.8f;
        estimate.reasoning = "Medium number - typical factor distribution";
    } else {
        estimate.confidence = 0.6f;
        estimate.reasoning = "Large number - factor size less predictable";
    }
    
    // Ensure bounds are reasonable
    estimate.min_digits = std::max(estimate.min_digits, 1);
    estimate.max_digits = std::min(estimate.max_digits, remaining_digits);
    
    return estimate;
}

//=============================================================================
// Special Form Detection
//=============================================================================

/**
 * Check if number is a Mersenne number (2^p - 1)
 */
bool is_mersenne_number(const factor_t* number, int* exponent) {
    factor_t n_plus_1 = add_128(*number, factor_t(1));
    
    // Count trailing zeros (which is the exponent p)
    int p = 0;
    factor_t temp = n_plus_1;
    
    while ((temp.low & 1) == 0 && !temp.is_zero()) {
        temp = shift_right_128(temp, 1);
        p++;
    }
    
    // Check if remaining is 1 (meaning n+1 = 2^p)
    if (temp == factor_t(1)) {
        if (exponent) *exponent = p;
        return true;
    }
    
    return false;
}

/**
 * Check if number is a Fermat number (2^(2^n) + 1)
 */
bool is_fermat_number(const factor_t* number, int* index) {
    factor_t n_minus_1 = subtract_128(*number, factor_t(1));
    
    // Count trailing zeros
    int zeros = 0;
    factor_t temp = n_minus_1;
    
    while ((temp.low & 1) == 0 && !temp.is_zero()) {
        temp = shift_right_128(temp, 1);
        zeros++;
    }
    
    // Check if zeros is a power of 2 and remaining is 1
    if (temp == factor_t(1) && zeros > 0) {
        // Check if zeros is 2^n
        int n = 0;
        int power = 1;
        while (power < zeros) {
            power <<= 1;
            n++;
        }
        
        if (power == zeros) {
            if (index) *index = n;
            return true;
        }
    }
    
    return false;
}

/**
 * Check if number is a perfect power (a^k for k > 1)
 */
bool is_perfect_power(const factor_t* number, factor_t* base, int* exponent) {
    // Check for small exponents (2 to 10)
    for (int k = 2; k <= 10; k++) {
        // Binary search for k-th root
        factor_t low(1);
        factor_t high = *number;
        
        while (low < high) {
            // mid = (low + high) / 2
            factor_t mid = shift_right_128(add_128(low, high), 1);
            
            // Compute mid^k
            factor_t power = mid;
            for (int i = 1; i < k; i++) {
                power = multiply_128_128(power, mid).low_128();
                if (power > *number) break;
            }
            
            if (power == *number) {
                if (base) *base = mid;
                if (exponent) *exponent = k;
                return true;
            } else if (power < *number) {
                low = add_128(mid, factor_t(1));
            } else {
                high = mid;
            }
        }
    }
    
    return false;
}

#endif // FACTORIZER_V22_SELECTOR_CUH