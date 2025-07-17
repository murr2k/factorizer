/**
 * Pollard's Rho with cuRAND Integration
 * High-quality randomness for CUDA warp-level parallelism
 */

#ifndef CURAND_POLLARDS_RHO_CUH
#define CURAND_POLLARDS_RHO_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "uint128_improved.cuh"
#include "barrett_reduction.cuh"

#define MAX_FACTORS 32
#define THREADS_PER_BLOCK 256
#define DEFAULT_MAX_ITERATIONS 1000000

// Pollard's Rho state per thread
struct PollardsRhoState {
    uint128_t x;
    uint128_t y;
    uint128_t c;  // Random constant for f(x) = x^2 + c
    curandState_t rand_state;
    Barrett128 barrett;
    int iteration;
};

// Initialize cuRAND state for each thread
__device__ void init_curand_state(curandState_t* state, int tid) {
    // Use clock64() for additional entropy beyond thread ID
    unsigned long long seed = clock64() + tid * 1337;
    curand_init(seed, tid, 0, state);
}

// Generate random 128-bit number using cuRAND
__device__ uint128_t generate_random_128(curandState_t* state, const uint128_t& max) {
    // Generate two 64-bit random values
    uint64_t low = curand(state);
    uint64_t high = curand(state);
    
    // Combine into 128-bit and reduce modulo max
    uint128_t rand_val(low, high);
    
    // Simple modulo for now - could use Barrett here too
    if (max.high == 0) {
        rand_val.high = 0;
        rand_val.low = rand_val.low % max.low;
    }
    
    return rand_val;
}

// Pollard's f function: f(x) = (x^2 + c) mod n
__device__ uint128_t pollards_f(
    const uint128_t& x, 
    const uint128_t& c, 
    const Barrett128& barrett
) {
    // x^2 mod n
    uint256_t x_squared = multiply_128_128(x, x);
    uint128_t result = barrett.reduce(uint128_t(x_squared.word[0], x_squared.word[1]));
    
    // Add c
    result = add_128(result, c);
    
    // Final reduction if needed
    if (result >= barrett.n) {
        result = subtract_128(result, barrett.n);
    }
    
    return result;
}

// Main Pollard's Rho kernel with cuRAND
__global__ void pollards_rho_curand(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    int max_iterations = DEFAULT_MAX_ITERATIONS
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;  // Lane within warp
    
    // Initialize state
    PollardsRhoState state;
    init_curand_state(&state.rand_state, tid);
    
    // Initialize Barrett reduction
    state.barrett.n = n;
    state.barrett.precompute();
    
    // Random starting point and constant
    state.x = generate_random_128(&state.rand_state, n);
    state.y = state.x;
    state.c = uint128_t(curand(&state.rand_state) % 100 + 1, 0);  // c in [1, 100]
    
    uint128_t factor(1, 0);
    state.iteration = 0;
    
    // Main Pollard's Rho loop
    while (factor.low == 1 && factor.high == 0 && state.iteration < max_iterations) {
        // Tortoise step: x = f(x)
        state.x = pollards_f(state.x, state.c, state.barrett);
        
        // Hare step: y = f(f(y))
        state.y = pollards_f(state.y, state.c, state.barrett);
        state.y = pollards_f(state.y, state.c, state.barrett);
        
        // Calculate |x - y|
        uint128_t diff;
        if (state.x > state.y) {
            diff = subtract_128(state.x, state.y);
        } else {
            diff = subtract_128(state.y, state.x);
        }
        
        // Calculate GCD
        factor = gcd_128(diff, n);
        
        state.iteration++;
        
        // Warp-level collaboration: check if any thread found a factor
        unsigned mask = __ballot_sync(0xFFFFFFFF, factor.low > 1 || factor.high > 0);
        
        if (mask != 0) {
            // At least one thread found a factor
            int source_lane = __ffs(mask) - 1;
            
            // Broadcast the factor to all threads in warp
            factor.low = __shfl_sync(0xFFFFFFFF, factor.low, source_lane);
            factor.high = __shfl_sync(0xFFFFFFFF, factor.high, source_lane);
            
            break;
        }
        
        // Periodic re-randomization to avoid cycles
        if (state.iteration % 1000 == 0) {
            state.c = uint128_t(curand(&state.rand_state) % 100 + 1, 0);
        }
    }
    
    // Store factor if found and non-trivial
    if ((factor.low > 1 || factor.high > 0) && factor < n) {
        // Use atomic to safely increment counter
        int idx = atomicAdd(factor_count, 1);
        if (idx < MAX_FACTORS) {
            factors[idx] = factor;
            
            // Also compute and store the cofactor
            uint128_t cofactor = divide_128_64(n, factor.low);  // Simplified for now
            if (cofactor.low > 1 || cofactor.high > 0) {
                int idx2 = atomicAdd(factor_count, 1);
                if (idx2 < MAX_FACTORS) {
                    factors[idx2] = cofactor;
                }
            }
        }
    }
}

// Advanced Pollard's Rho with Brent's improvement
__global__ void pollards_rho_brent(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    int max_iterations = DEFAULT_MAX_ITERATIONS
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize cuRAND
    curandState_t rand_state;
    init_curand_state(&rand_state, tid);
    
    // Barrett reduction setup
    Barrett128 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // Brent's algorithm variables
    uint128_t y = generate_random_128(&rand_state, n);
    uint128_t c = uint128_t(curand(&rand_state) % 100 + 1, 0);
    uint128_t m = uint128_t(curand(&rand_state) % 100 + 1, 0);
    
    uint128_t factor(1, 0);
    uint128_t r(1, 0);
    uint128_t q(1, 0);
    
    uint128_t x, ys;
    
    do {
        x = y;
        for (uint64_t i = 0; i < r.low; i++) {
            y = pollards_f(y, c, barrett);
        }
        
        uint128_t k(0, 0);
        while (k < r && factor.low == 1 && factor.high == 0) {
            ys = y;
            
            // Inner loop with product accumulation
            uint128_t min_val = m;
            if (subtract_128(r, k) < m) {
                min_val = subtract_128(r, k);
            }
            
            for (uint64_t i = 0; i < min_val.low; i++) {
                y = pollards_f(y, c, barrett);
                
                // Accumulate product
                uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
                uint256_t prod = multiply_128_128(q, diff);
                q = barrett.reduce(uint128_t(prod.word[0], prod.word[1]));
            }
            
            factor = gcd_128(q, n);
            k = add_128(k, m);
        }
        
        r = shift_left_128(r, 1);  // r *= 2
        
    } while (factor.low == 1 && factor.high == 0);
    
    // Backtrack if necessary
    if (factor == n) {
        do {
            ys = pollards_f(ys, c, barrett);
            uint128_t diff = (x > ys) ? subtract_128(x, ys) : subtract_128(ys, x);
            factor = gcd_128(diff, n);
        } while (factor.low == 1 && factor.high == 0);
    }
    
    // Store factor if found
    if ((factor.low > 1 || factor.high > 0) && factor < n) {
        int idx = atomicAdd(factor_count, 1);
        if (idx < MAX_FACTORS) {
            factors[idx] = factor;
        }
    }
}

// Test kernel for cuRAND integration
__global__ void test_curand_pollards() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Testing cuRAND Pollard's Rho implementation\n");
        
        // Test case: 90595490423 = 428759 × 211297
        uint128_t n(90595490423ULL, 0);
        uint128_t factors[MAX_FACTORS];
        int factor_count = 0;
        
        // Note: This is a simplified test - real implementation would 
        // launch this as a separate kernel
        printf("Test number: %llu\n", n.low);
        printf("Expected factors: 428759 and 211297\n");
        
        // Test random number generation
        curandState_t state;
        init_curand_state(&state, 0);
        
        for (int i = 0; i < 5; i++) {
            uint128_t rand = generate_random_128(&state, n);
            printf("Random %d: %llu\n", i, rand.low);
        }
    }
}

// Utility function to launch Pollard's Rho
void launch_pollards_rho(
    uint128_t n,
    uint128_t* d_factors,
    int* d_factor_count,
    int num_blocks = 32,
    int threads_per_block = THREADS_PER_BLOCK,
    int max_iterations = DEFAULT_MAX_ITERATIONS
) {
    // Reset factor count
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    // Launch kernel
    pollards_rho_curand<<<num_blocks, threads_per_block>>>(
        n, d_factors, d_factor_count, max_iterations
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

#endif // CURAND_POLLARDS_RHO_CUH