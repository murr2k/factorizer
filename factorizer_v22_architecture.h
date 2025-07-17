/**
 * Factorizer v2.2.0 Architecture Header
 * 
 * Modular architecture supporting multiple factorization algorithms
 * with intelligent selection and GPU optimization.
 * 
 * Design Goals:
 * - Support numbers with factors up to 20 digits
 * - Clean interfaces between components
 * - Extensible algorithm framework
 * - Efficient memory management
 * - Progress tracking for long operations
 */

#ifndef FACTORIZER_V22_ARCHITECTURE_H
#define FACTORIZER_V22_ARCHITECTURE_H

#include <stdint.h>
#include <cuda_runtime.h>
#include "uint128_improved.cuh"

// Forward declarations
struct factorizer_context;
struct algorithm_stats;
struct memory_pool;
struct progress_tracker;

//=============================================================================
// Core Type Definitions
//=============================================================================

// Extended precision integer types
typedef uint128_t factor_t;
typedef uint256_t factor_product_t;

// Algorithm identifiers
typedef enum {
    ALGO_AUTO           = 0,    // Automatic selection
    ALGO_TRIAL_DIVISION = 1,    // For small factors
    ALGO_POLLARD_RHO    = 2,    // General purpose
    ALGO_POLLARD_P1     = 3,    // For factors p where p-1 is smooth
    ALGO_ECM            = 4,    // Elliptic Curve Method
    ALGO_QUADRATIC_SIEVE = 5,   // For medium numbers
    ALGO_GNFS           = 6,    // General Number Field Sieve (future)
    ALGO_HYBRID         = 7,    // Combined approach
    ALGO_COUNT
} algorithm_id_t;

// Algorithm capabilities and characteristics
typedef struct {
    algorithm_id_t id;
    const char* name;
    int min_digit_efficient;    // Minimum digits for efficiency
    int max_digit_efficient;    // Maximum digits for efficiency
    int gpu_memory_mb;          // Typical GPU memory requirement
    float complexity_factor;    // Relative computational complexity
    bool supports_partial;      // Can find partial factors
    bool requires_smoothness;   // Needs smooth numbers
} algorithm_info_t;

// Number characteristics for algorithm selection
typedef struct {
    int digit_count;            // Number of decimal digits
    int bit_count;              // Number of bits
    bool is_probable_prime;     // Primality test result
    int small_factor_bound;     // Largest small factor checked
    float smoothness_estimate;  // Estimated smoothness parameter
    bool has_special_form;      // Mersenne, Fermat, etc.
    uint64_t trailing_zeros;    // Power of 2 factor
} number_characteristics_t;

//=============================================================================
// Memory Management
//=============================================================================

// Memory allocation types
typedef enum {
    MEM_TYPE_HOST,              // CPU memory
    MEM_TYPE_DEVICE,            // GPU global memory
    MEM_TYPE_UNIFIED,           // Unified memory
    MEM_TYPE_PINNED             // Pinned host memory
} memory_type_t;

// Memory block descriptor
typedef struct memory_block {
    void* ptr;
    size_t size;
    memory_type_t type;
    bool in_use;
    struct memory_block* next;
} memory_block_t;

// Memory pool for efficient allocation
struct memory_pool {
    memory_block_t* blocks[4];  // One list per memory type
    size_t total_allocated[4];
    size_t high_water_mark[4];
    cudaStream_t stream;        // Associated CUDA stream
};

// Memory management functions
typedef struct {
    void* (*allocate)(memory_pool_t* pool, size_t size, memory_type_t type);
    void (*free)(memory_pool_t* pool, void* ptr);
    void (*reset)(memory_pool_t* pool);
    void (*destroy)(memory_pool_t* pool);
    size_t (*get_usage)(memory_pool_t* pool, memory_type_t type);
} memory_manager_t;

//=============================================================================
// Progress Tracking
//=============================================================================

// Progress information
typedef struct {
    uint64_t iterations_completed;
    uint64_t iterations_total;      // Estimated total
    float percentage;               // 0.0 to 100.0
    double elapsed_seconds;
    double estimated_remaining;
    algorithm_id_t current_algorithm;
    int factors_found;
    char status_message[256];
} progress_info_t;

// Progress tracker
struct progress_tracker {
    progress_info_t info;
    volatile int* device_counter;   // GPU progress counter
    cudaEvent_t start_event;
    cudaEvent_t current_event;
    void (*callback)(const progress_info_t* info, void* user_data);
    void* callback_data;
    int update_interval_ms;
};

//=============================================================================
// Algorithm Interface
//=============================================================================

// Algorithm parameters
typedef struct {
    factor_t number;                // Number to factor
    int max_iterations;             // Maximum iterations
    int thread_count;               // GPU threads
    int block_count;                // GPU blocks
    memory_pool_t* memory_pool;     // Memory allocator
    progress_tracker_t* tracker;    // Progress tracking
    void* algorithm_specific;       // Algorithm-specific params
} algorithm_params_t;

// Algorithm result
typedef struct {
    factor_t factors[64];           // Found factors
    int factor_count;               // Number of factors found
    bool is_complete;               // All factors found
    uint64_t iterations;            // Total iterations
    double runtime_seconds;         // Execution time
    algorithm_stats_t* stats;       // Detailed statistics
} algorithm_result_t;

// Algorithm interface - all algorithms implement this
typedef struct {
    algorithm_id_t id;
    const char* name;
    
    // Initialize algorithm-specific resources
    int (*initialize)(memory_pool_t* pool, void** context);
    
    // Clean up resources
    void (*cleanup)(void* context);
    
    // Estimate runtime and memory for given number
    void (*estimate_requirements)(
        const factor_t* number,
        const number_characteristics_t* chars,
        size_t* memory_bytes,
        double* estimated_seconds
    );
    
    // Run the factorization algorithm
    int (*factor)(
        const algorithm_params_t* params,
        algorithm_result_t* result
    );
    
    // Check if algorithm is suitable for number
    float (*suitability_score)(
        const factor_t* number,
        const number_characteristics_t* chars
    );
} algorithm_interface_t;

//=============================================================================
// Algorithm Statistics
//=============================================================================

struct algorithm_stats {
    // Performance metrics
    uint64_t total_iterations;
    uint64_t useful_iterations;     // Iterations that made progress
    double gpu_utilization;         // 0.0 to 1.0
    double memory_bandwidth_gbps;
    
    // Algorithm-specific metrics
    union {
        struct {
            uint64_t gcd_calls;
            uint64_t collision_count;
            double avg_cycle_length;
        } pollard_rho;
        
        struct {
            uint64_t smooth_numbers_found;
            uint64_t matrix_dimension;
            double sieving_efficiency;
        } quadratic_sieve;
        
        struct {
            uint64_t curves_tried;
            uint64_t stage1_points;
            uint64_t stage2_points;
        } ecm;
    } algorithm_specific;
};

//=============================================================================
// Factor Size Estimation
//=============================================================================

// Estimate the size of smallest remaining factor
typedef struct {
    int min_digits;                 // Minimum digit count
    int max_digits;                 // Maximum digit count
    float confidence;               // Confidence level (0.0 to 1.0)
    const char* reasoning;          // Explanation
} factor_size_estimate_t;

// Factor size estimation functions
factor_size_estimate_t estimate_factor_size(
    const factor_t* number,
    const factor_t* known_factors,
    int known_count
);

// Check for special number forms
bool is_mersenne_number(const factor_t* number, int* exponent);
bool is_fermat_number(const factor_t* number, int* index);
bool is_perfect_power(const factor_t* number, factor_t* base, int* exponent);

//=============================================================================
// Algorithm Selector
//=============================================================================

// Selection criteria
typedef struct {
    bool prefer_speed;              // Optimize for speed vs thoroughness
    bool allow_probabilistic;       // Allow algorithms that might miss factors
    size_t memory_limit_mb;         // GPU memory constraint
    double time_limit_seconds;      // Maximum runtime
    algorithm_id_t preferred;       // User preference
    algorithm_id_t exclude[ALGO_COUNT]; // Algorithms to exclude
} selection_criteria_t;

// Algorithm selector interface
typedef struct {
    // Select best algorithm for number
    algorithm_id_t (*select_algorithm)(
        const factor_t* number,
        const number_characteristics_t* chars,
        const selection_criteria_t* criteria
    );
    
    // Get ranked list of algorithms
    void (*rank_algorithms)(
        const factor_t* number,
        const number_characteristics_t* chars,
        algorithm_id_t* ranking,
        float* scores,
        int count
    );
    
    // Update selection model with results
    void (*update_model)(
        const factor_t* number,
        algorithm_id_t algorithm,
        const algorithm_result_t* result
    );
} algorithm_selector_t;

//=============================================================================
// Main Factorizer Context
//=============================================================================

struct factorizer_context {
    // Configuration
    int device_id;                  // CUDA device
    cudaDeviceProp device_props;    // Device properties
    
    // Resources
    memory_pool_t* memory_pool;     // Memory manager
    algorithm_interface_t* algorithms[ALGO_COUNT]; // Available algorithms
    algorithm_selector_t* selector; // Algorithm selector
    
    // State
    factor_t current_number;        // Number being factored
    algorithm_id_t current_algorithm; // Active algorithm
    progress_tracker_t* tracker;    // Progress tracking
    
    // Results cache
    struct {
        factor_t number;
        factor_t factors[64];
        int count;
        bool valid;
    } cache;
};

//=============================================================================
// Public API
//=============================================================================

// Initialize factorizer context
factorizer_context_t* factorizer_create(int device_id);

// Destroy context and free resources
void factorizer_destroy(factorizer_context_t* ctx);

// Analyze number characteristics
void factorizer_analyze_number(
    factorizer_context_t* ctx,
    const factor_t* number,
    number_characteristics_t* chars
);

// Factor a number with automatic algorithm selection
int factorizer_factor_auto(
    factorizer_context_t* ctx,
    const factor_t* number,
    factor_t* factors,
    int* factor_count,
    progress_tracker_t* tracker
);

// Factor using specific algorithm
int factorizer_factor_with_algorithm(
    factorizer_context_t* ctx,
    const factor_t* number,
    algorithm_id_t algorithm,
    factor_t* factors,
    int* factor_count,
    progress_tracker_t* tracker
);

// Get algorithm information
const algorithm_info_t* factorizer_get_algorithm_info(algorithm_id_t id);

// Utility functions
int factorizer_parse_number(const char* str, factor_t* number);
void factorizer_print_number(const factor_t* number, char* buffer, size_t size);
void factorizer_print_progress(const progress_info_t* info);

//=============================================================================
// Error Codes
//=============================================================================

#define FACTORIZER_SUCCESS              0
#define FACTORIZER_ERROR_INVALID_INPUT  -1
#define FACTORIZER_ERROR_NO_MEMORY      -2
#define FACTORIZER_ERROR_CUDA           -3
#define FACTORIZER_ERROR_TIMEOUT        -4
#define FACTORIZER_ERROR_NOT_IMPLEMENTED -5
#define FACTORIZER_ERROR_ALGORITHM      -6

//=============================================================================
// Algorithm Registration Macros
//=============================================================================

#define REGISTER_ALGORITHM(id, impl) \
    do { \
        extern algorithm_interface_t impl; \
        ctx->algorithms[id] = &impl; \
    } while(0)

#define DEFINE_ALGORITHM(name) \
    algorithm_interface_t algorithm_##name = { \
        .id = ALGO_##name, \
        .name = #name, \
        .initialize = name##_initialize, \
        .cleanup = name##_cleanup, \
        .estimate_requirements = name##_estimate, \
        .factor = name##_factor, \
        .suitability_score = name##_suitability \
    }

#endif // FACTORIZER_V22_ARCHITECTURE_H