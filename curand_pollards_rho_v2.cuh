/**
 * Pollard's Rho with cuRAND Integration v2.0
 * Production-ready with error handling and optimizations
 * High-quality randomness for CUDA warp-level parallelism
 */

#ifndef CURAND_POLLARDS_RHO_V2_CUH
#define CURAND_POLLARDS_RHO_V2_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"
#include "montgomery_reduction.cuh"

#define MAX_FACTORS 32
#define THREADS_PER_BLOCK 256
#define DEFAULT_MAX_ITERATIONS 1000000
#define WARP_SIZE 32

// Error codes for cuRAND operations
enum CurandError {
    CURAND_SUCCESS = 0,
    CURAND_INIT_FAILED = 1,
    CURAND_GENERATION_FAILED = 2,
    CURAND_INVALID_PARAMETER = 3
};

// Enhanced Pollard's Rho state with error tracking
struct PollardsRhoState_v2 {
    uint128_t x;
    uint128_t y;
    uint128_t c;                // Random constant for f(x) = x^2 + c
    curandState_t rand_state;
    Barrett128_v2 barrett;       // Use improved Barrett
    Montgomery128 montgomery;    // Option for Montgomery
    int iteration;
    int error_code;
    bool use_montgomery;         // Flag to choose reduction method
};

// Result structure for better error reporting
struct FactorizationResult {
    uint128_t factors[MAX_FACTORS];
    int factor_count;
    int total_iterations;
    int successful_threads;
    int error_count;
    double time_ms;
};

// Safe cuRAND initialization with error checking
__device__ bool init_curand_state_safe(curandState_t* state, int tid, int* error_code) {
    // Use multiple entropy sources
    unsigned long long seed = 0;
    
    // Try to use clock64() if available
    #if __CUDA_ARCH__ >= 200
    seed = clock64();
    #else
    seed = clock();
    #endif
    
    // Add thread ID and block ID for uniqueness
    seed += tid * 1337 + blockIdx.x * 7919;
    
    // Add grid dimensions for more entropy
    seed ^= (gridDim.x * blockDim.x) * 31;
    
    // Initialize cuRAND
    curand_init(seed, tid, 0, state);
    
    // Verify initialization by generating a test number
    unsigned int test = curand(state);
    if (test == 0 && curand(state) == 0) {
        *error_code = CURAND_INIT_FAILED;
        return false;
    }
    
    *error_code = CURAND_SUCCESS;
    return true;
}

// Optimized random 128-bit generation with bounds checking
__device__ uint128_t generate_random_128_safe(
    curandState_t* state, 
    const uint128_t& min,
    const uint128_t& max,
    int* error_code
) {
    if (max <= min) {
        *error_code = CURAND_INVALID_PARAMETER;
        return min;
    }
    
    // Generate two 64-bit random values
    uint64_t low = curand(state);
    uint64_t high = curand(state);
    
    // Handle special cases for better distribution
    uint128_t range = subtract_128(max, min);
    
    if (range.high == 0) {
        // Range fits in 64 bits - use only low part for better distribution
        low = low % range.low;
        high = 0;
    } else {
        // Full 128-bit range - use rejection sampling for uniform distribution
        uint128_t rand_val(low, high);
        
        // Simple modulo reduction (could be improved with Barrett)
        while (rand_val >= range) {
            low = curand(state);
            high = curand(state);
            rand_val = uint128_t(low, high);
        }
        
        return add_128(min, rand_val);
    }
    
    *error_code = CURAND_SUCCESS;
    return add_128(min, uint128_t(low, high));
}

// Pollard's f function with choice of reduction method
__device__ uint128_t pollards_f_optimized(
    const uint128_t& x, 
    const uint128_t& c, 
    const PollardsRhoState_v2& state
) {
    // x^2 mod n
    uint256_t x_squared = multiply_128_128(x, x);
    
    uint128_t result;
    if (state.use_montgomery) {
        // Use Montgomery reduction (if x is in Montgomery form)
        result = montgomery_reduce(x_squared, state.montgomery);
    } else {
        // Use Barrett reduction
        result = state.barrett.reduce(x_squared);
    }
    
    // Add c
    result = add_128(result, c);
    
    // Final reduction if needed
    if (result >= state.barrett.n) {
        result = subtract_128(result, state.barrett.n);
    }
    
    return result;
}

// Brent's cycle detection optimization
__device__ uint128_t brents_cycle_detection(
    PollardsRhoState_v2& state,
    const uint128_t& n,
    int max_iterations
) {
    uint128_t power = uint128_t(1, 0);
    uint128_t lam = uint128_t(1, 0);  // λ (cycle length)
    uint128_t tortoise = state.x;
    uint128_t hare = state.x;
    
    // Find cycle length
    do {
        if (power == lam) {
            tortoise = hare;
            power = shift_left_128(power, 1);
            lam = uint128_t(0, 0);
        }
        hare = pollards_f_optimized(hare, state.c, state);
        lam = add_128(lam, uint128_t(1, 0));
        state.iteration++;
    } while (tortoise != hare && state.iteration < max_iterations);
    
    // Find position of first repetition
    uint128_t mu(0, 0);  // μ (cycle start position)
    tortoise = state.x;
    hare = state.x;
    
    for (uint64_t i = 0; i < lam.low; i++) {
        hare = pollards_f_optimized(hare, state.c, state);
    }
    
    // Find the GCD
    uint128_t factor(1, 0);
    while (tortoise != hare && factor.low == 1) {
        tortoise = pollards_f_optimized(tortoise, state.c, state);
        hare = pollards_f_optimized(hare, state.c, state);
        
        uint128_t diff = (tortoise > hare) ? 
            subtract_128(tortoise, hare) : 
            subtract_128(hare, tortoise);
        
        factor = gcd_128(diff, n);
        mu = add_128(mu, uint128_t(1, 0));
        state.iteration++;
    }
    
    return factor;
}

// Main Pollard's Rho kernel with enhanced error handling
__global__ void pollards_rho_curand_v2(
    uint128_t n,
    FactorizationResult* result,
    int max_iterations = DEFAULT_MAX_ITERATIONS,
    bool use_montgomery = false,
    bool use_brent = false
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid & (WARP_SIZE - 1);
    
    // Initialize state
    PollardsRhoState_v2 state;
    state.iteration = 0;
    state.error_code = CURAND_SUCCESS;
    state.use_montgomery = use_montgomery && (n.low & 1);  // Montgomery requires odd n
    
    // Initialize random state with error checking
    if (!init_curand_state_safe(&state.rand_state, tid, &state.error_code)) {
        atomicAdd(&result->error_count, 1);
        return;
    }
    
    // Initialize reduction method
    if (state.use_montgomery) {
        state.montgomery.n = n;
        state.montgomery.precompute();
    } else {
        state.barrett.n = n;
        state.barrett.precompute();
    }
    
    // Random starting point and constant
    state.x = generate_random_128_safe(&state.rand_state, uint128_t(2, 0), n, &state.error_code);
    state.y = state.x;
    state.c = generate_random_128_safe(&state.rand_state, uint128_t(1, 0), uint128_t(100, 0), &state.error_code);
    
    if (state.error_code != CURAND_SUCCESS) {
        atomicAdd(&result->error_count, 1);
        return;
    }
    
    uint128_t factor(1, 0);
    
    // Choose algorithm variant
    if (use_brent) {
        factor = brents_cycle_detection(state, n, max_iterations);
    } else {
        // Standard Pollard's Rho
        while (factor.low == 1 && factor.high == 0 && state.iteration < max_iterations) {
            // Tortoise step
            state.x = pollards_f_optimized(state.x, state.c, state);
            
            // Hare steps
            state.y = pollards_f_optimized(state.y, state.c, state);
            state.y = pollards_f_optimized(state.y, state.c, state);
            
            // Calculate |x - y|
            uint128_t diff = (state.x > state.y) ? 
                subtract_128(state.x, state.y) : 
                subtract_128(state.y, state.x);
            
            // Calculate GCD
            factor = gcd_128(diff, n);
            state.iteration++;
            
            // Warp-level collaboration
            unsigned mask = __ballot_sync(0xFFFFFFFF, factor.low > 1 || factor.high > 0);
            
            if (mask != 0) {
                int source_lane = __ffs(mask) - 1;
                factor.low = __shfl_sync(0xFFFFFFFF, factor.low, source_lane);
                factor.high = __shfl_sync(0xFFFFFFFF, factor.high, source_lane);
                break;
            }
            
            // Adaptive re-randomization
            if (state.iteration % (1000 + tid % 100) == 0) {
                state.c = generate_random_128_safe(&state.rand_state, uint128_t(1, 0), uint128_t(100, 0), &state.error_code);
                if (state.error_code != CURAND_SUCCESS) break;
            }
        }
    }
    
    // Store factor if found and non-trivial
    if ((factor.low > 1 || factor.high > 0) && factor < n && factor != n) {
        int idx = atomicAdd(&result->factor_count, 1);
        if (idx < MAX_FACTORS) {
            result->factors[idx] = factor;
            
            // Try to find cofactor
            if (n.high == 0 && factor.high == 0 && factor.low != 0) {
                uint128_t cofactor(n.low / factor.low, 0);
                if (cofactor.low > 1) {
                    int idx2 = atomicAdd(&result->factor_count, 1);
                    if (idx2 < MAX_FACTORS) {
                        result->factors[idx2] = cofactor;
                    }
                }
            }
        }
        atomicAdd(&result->successful_threads, 1);
    }
    
    // Update total iterations
    atomicAdd(&result->total_iterations, state.iteration);
}

// Optimized kernel launcher with automatic parameter tuning
void launch_pollards_rho_v2(
    uint128_t n,
    FactorizationResult* h_result,
    int num_blocks = 0,
    int threads_per_block = THREADS_PER_BLOCK,
    int max_iterations = DEFAULT_MAX_ITERATIONS,
    bool use_montgomery = false,
    bool use_brent = false
) {
    // Auto-tune grid size if not specified
    if (num_blocks == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Use multiple of SM count for better load balancing
        num_blocks = prop.multiProcessorCount * 2;
        
        // Adjust based on problem size
        if (n.high > 0) {
            num_blocks *= 2;  // More threads for larger numbers
        }
    }
    
    // Allocate device memory for results
    FactorizationResult* d_result;
    cudaMalloc(&d_result, sizeof(FactorizationResult));
    cudaMemset(d_result, 0, sizeof(FactorizationResult));
    
    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch kernel
    pollards_rho_curand_v2<<<num_blocks, threads_per_block>>>(
        n, d_result, max_iterations, use_montgomery, use_brent
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        h_result->error_count = -1;
        return;
    }
    
    // Wait for completion and measure time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back
    cudaMemcpy(h_result, d_result, sizeof(FactorizationResult), cudaMemcpyDeviceToHost);
    h_result->time_ms = milliseconds;
    
    // Cleanup
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Test kernel for cuRAND v2
__global__ void test_curand_v2() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Testing cuRAND v2 with error handling\n");
        
        // Test initialization
        curandState_t state;
        int error_code;
        
        if (init_curand_state_safe(&state, 0, &error_code)) {
            printf("cuRAND initialization: SUCCESS\n");
            
            // Test random generation
            uint128_t min(10, 0);
            uint128_t max(1000, 0);
            
            printf("Generating 5 random numbers in range [10, 1000]:\n");
            for (int i = 0; i < 5; i++) {
                uint128_t rand = generate_random_128_safe(&state, min, max, &error_code);
                if (error_code == CURAND_SUCCESS) {
                    printf("  Random %d: %llu\n", i, rand.low);
                } else {
                    printf("  Random %d: ERROR (code %d)\n", i, error_code);
                }
            }
        } else {
            printf("cuRAND initialization: FAILED (error code %d)\n", error_code);
        }
    }
}

#endif // CURAND_POLLARDS_RHO_V2_CUH