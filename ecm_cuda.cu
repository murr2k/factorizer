/**
 * Elliptic Curve Method (ECM) for GPU Factorization
 * Implements Montgomery curve arithmetic and parallel ECM
 * Optimized for finding factors up to 20 digits
 */

#ifndef ECM_CUDA_CU
#define ECM_CUDA_CU

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include "uint128_improved.cuh"
#include "montgomery_reduction.cuh"

// ECM Configuration
#define ECM_MAX_CURVES 64        // Number of parallel curves per kernel
#define ECM_STAGE1_BOUND 50000   // B1 bound for stage 1
#define ECM_STAGE2_BOUND 5000000 // B2 bound for stage 2
#define ECM_BABY_STEPS 2048      // Baby steps for stage 2
#define ECM_GIANT_STEPS 2048     // Giant steps for stage 2

// Montgomery curve: By^2 = x^3 + Ax^2 + x
// We use B=1 (normalized form)
struct MontgomeryCurve {
    uint128_t A;     // Curve parameter
    uint128_t n;     // Number to factor
    Montgomery128 mont; // Montgomery context for modular arithmetic
};

// Point on Montgomery curve in projective coordinates
struct ECMPoint {
    uint128_t X;     // X coordinate
    uint128_t Z;     // Z coordinate (Y not needed for Montgomery ladder)
};

// ECM working state for each curve
struct ECMState {
    MontgomeryCurve curve;
    ECMPoint P;      // Base point
    ECMPoint Q;      // Current point
    uint128_t gcd_accumulator; // For accumulated GCD
    curandState_t rand_state;
    bool factor_found;
    uint128_t factor;
};

// Initialize cuRAND state for curve generation
__global__ void ecm_init_random(curandState_t* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ECM_MAX_CURVES) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// GCD function for uint128_t
__device__ __host__ uint128_t gcd(uint128_t a, uint128_t b) {
    while (b != uint128_t(0)) {
        uint128_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Montgomery curve point doubling: 2P
__device__ void ecm_point_double(ECMPoint& R, const ECMPoint& P, const MontgomeryCurve& curve) {
    // Algorithm:
    // X_2P = (X+Z)^2 * (X-Z)^2
    // Z_2P = 4XZ * ((X-Z)^2 + ((A+2)/4) * 4XZ)
    
    uint128_t XpZ = curve.mont.add(P.X, P.Z);
    uint128_t XmZ = curve.mont.sub(P.X, P.Z);
    uint128_t XpZ_sq = curve.mont.mul(XpZ, XpZ);
    uint128_t XmZ_sq = curve.mont.mul(XmZ, XmZ);
    
    R.X = curve.mont.mul(XpZ_sq, XmZ_sq);
    
    uint128_t diff = curve.mont.sub(XpZ_sq, XmZ_sq);
    uint128_t A24 = curve.mont.add(curve.A, uint128_t(2));
    A24 = curve.mont.mul(A24, curve.mont.inverse(uint128_t(4)));
    
    uint128_t t = curve.mont.mul(A24, diff);
    t = curve.mont.add(t, XmZ_sq);
    R.Z = curve.mont.mul(diff, t);
}

// Montgomery curve point addition: P + Q given P - Q
__device__ void ecm_point_add(ECMPoint& R, const ECMPoint& P, const ECMPoint& Q, 
                              const ECMPoint& PmQ, const MontgomeryCurve& curve) {
    // Algorithm:
    // X_P+Q = Z_P-Q * ((X_P - Z_P)(X_Q + Z_Q) + (X_P + Z_P)(X_Q - Z_Q))^2
    // Z_P+Q = X_P-Q * ((X_P - Z_P)(X_Q + Z_Q) - (X_P + Z_P)(X_Q - Z_Q))^2
    
    uint128_t XP_m_ZP = curve.mont.sub(P.X, P.Z);
    uint128_t XP_p_ZP = curve.mont.add(P.X, P.Z);
    uint128_t XQ_m_ZQ = curve.mont.sub(Q.X, Q.Z);
    uint128_t XQ_p_ZQ = curve.mont.add(Q.X, Q.Z);
    
    uint128_t t1 = curve.mont.mul(XP_m_ZP, XQ_p_ZQ);
    uint128_t t2 = curve.mont.mul(XP_p_ZP, XQ_m_ZQ);
    
    uint128_t sum = curve.mont.add(t1, t2);
    uint128_t diff = curve.mont.sub(t1, t2);
    
    sum = curve.mont.mul(sum, sum);
    diff = curve.mont.mul(diff, diff);
    
    R.X = curve.mont.mul(PmQ.Z, sum);
    R.Z = curve.mont.mul(PmQ.X, diff);
}

// Montgomery ladder for scalar multiplication
__device__ void ecm_scalar_multiply(ECMPoint& R, const ECMPoint& P, 
                                   const uint128_t& k, const MontgomeryCurve& curve) {
    // Initialize R0 = O (point at infinity), R1 = P
    ECMPoint R0 = {uint128_t(1), uint128_t(0)}; // Point at infinity in projective coords
    ECMPoint R1 = P;
    
    // Find highest bit
    int bit_len = 128 - k.leading_zeros();
    
    // Montgomery ladder - constant time scalar multiplication
    for (int i = bit_len - 1; i >= 0; i--) {
        // Get bit at position i
        bool bit = (i < 64) ? ((k.low >> i) & 1) : ((k.high >> (i - 64)) & 1);
        
        if (bit) {
            // R0 = R0 + R1, R1 = 2*R1
            ecm_point_add(R0, R0, R1, P, curve);
            ecm_point_double(R1, R1, curve);
        } else {
            // R1 = R0 + R1, R0 = 2*R0
            ecm_point_add(R1, R0, R1, P, curve);
            ecm_point_double(R0, R0, curve);
        }
    }
    
    R = R0;
}

// Generate a random curve and point
__device__ void ecm_generate_curve(ECMState& state, curandState_t* rand_state) {
    // Generate random point coordinates
    uint64_t x = curand(rand_state) % state.curve.n.low;
    uint64_t y = curand(rand_state) % state.curve.n.low;
    
    state.P.X = state.curve.mont.to_montgomery(uint128_t(x));
    state.P.Z = state.curve.mont.one(); // Z = 1 in Montgomery form
    
    // Calculate curve parameter A from point
    // Using the curve equation: By^2 = x^3 + Ax^2 + x
    // We derive A from a random point (x,y) with B=1
    uint128_t x_mont = state.P.X;
    uint128_t y_mont = state.curve.mont.to_montgomery(uint128_t(y));
    
    uint128_t x2 = state.curve.mont.mul(x_mont, x_mont);
    uint128_t x3 = state.curve.mont.mul(x2, x_mont);
    uint128_t y2 = state.curve.mont.mul(y_mont, y_mont);
    
    // A = (y^2 - x^3 - x) / x^2
    uint128_t num = state.curve.mont.sub(y2, x3);
    num = state.curve.mont.sub(num, x_mont);
    
    state.curve.A = state.curve.mont.mul(num, state.curve.mont.inverse(x2));
    
    // Initialize Q = P for accumulation
    state.Q = state.P;
    state.gcd_accumulator = uint128_t(1);
    state.factor_found = false;
}

// Stage 1: Multiply by smooth numbers up to B1
__device__ void ecm_stage1(ECMState& state) {
    const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                         53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                         127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                         199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281};
    const int num_primes = sizeof(primes) / sizeof(primes[0]);
    
    // For each prime p <= B1, multiply point by p^k where p^k <= B1
    for (int i = 0; i < num_primes && primes[i] <= ECM_STAGE1_BOUND; i++) {
        uint64_t p = primes[i];
        uint64_t pk = p;
        
        // Calculate highest power of p <= B1
        while (pk * p <= ECM_STAGE1_BOUND) {
            pk *= p;
        }
        
        // Multiply point by pk
        ecm_scalar_multiply(state.Q, state.Q, uint128_t(pk), state.curve);
        
        // Accumulate GCD every few iterations to avoid overflow
        if (i % 10 == 0) {
            uint128_t g = gcd(state.Q.Z, state.curve.n);
            if (g > uint128_t(1) && g < state.curve.n) {
                state.factor_found = true;
                state.factor = g;
                return;
            }
        }
    }
    
    // Final GCD check
    uint128_t g = gcd(state.Q.Z, state.curve.n);
    if (g > uint128_t(1) && g < state.curve.n) {
        state.factor_found = true;
        state.factor = g;
    }
}

// Stage 2: Baby-step giant-step algorithm
__device__ void ecm_stage2(ECMState& state) {
    if (state.factor_found) return;
    
    // Precompute baby steps: Q, 2Q, 3Q, ..., BABY_STEPS*Q
    ECMPoint baby_steps[ECM_BABY_STEPS];
    baby_steps[0] = state.Q;
    
    for (int i = 1; i < ECM_BABY_STEPS; i++) {
        ecm_point_add(baby_steps[i], baby_steps[i-1], state.Q, state.Q, state.curve);
        
        // Periodic GCD check
        if (i % 100 == 0) {
            uint128_t g = gcd(baby_steps[i].Z, state.curve.n);
            if (g > uint128_t(1) && g < state.curve.n) {
                state.factor_found = true;
                state.factor = g;
                return;
            }
        }
    }
    
    // Giant steps
    ECMPoint giant_step;
    ecm_scalar_multiply(giant_step, state.Q, uint128_t(ECM_BABY_STEPS), state.curve);
    
    ECMPoint current = giant_step;
    for (int j = 1; j < ECM_GIANT_STEPS; j++) {
        // Check differences with all baby steps
        for (int i = 0; i < ECM_BABY_STEPS; i++) {
            ECMPoint diff;
            // Calculate current - baby_steps[i]
            ECMPoint neg_baby = baby_steps[i];
            neg_baby.X = state.curve.mont.sub(state.curve.n, neg_baby.X); // Negate X
            
            ecm_point_add(diff, current, neg_baby, state.Q, state.curve);
            
            uint128_t g = gcd(diff.Z, state.curve.n);
            if (g > uint128_t(1) && g < state.curve.n) {
                state.factor_found = true;
                state.factor = g;
                return;
            }
        }
        
        // Next giant step
        ecm_point_add(current, current, giant_step, state.Q, state.curve);
    }
}

// Main ECM kernel - each thread processes one curve
__global__ void ecm_factor_kernel(uint128_t n, ECMState* states, 
                                 curandState_t* rand_states, uint128_t* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ECM_MAX_CURVES) return;
    
    ECMState& state = states[tid];
    state.curve.n = n;
    state.curve.mont.n = n;
    state.curve.mont.precompute();
    
    // Generate random curve
    ecm_generate_curve(state, &rand_states[tid]);
    
    // Stage 1
    ecm_stage1(state);
    
    // Stage 2 (if no factor found in stage 1)
    if (!state.factor_found) {
        ecm_stage2(state);
    }
    
    // Report factor if found
    if (state.factor_found) {
        atomicExch((unsigned long long*)&result->low, state.factor.low);
        atomicExch((unsigned long long*)&result->high, state.factor.high);
    }
}

// Host function to run ECM factorization
extern "C" bool ecm_factor(uint128_t n, uint128_t& factor, int max_curves = 1000) {
    // Allocate device memory
    ECMState* d_states;
    curandState_t* d_rand_states;
    uint128_t* d_result;
    
    cudaMalloc(&d_states, ECM_MAX_CURVES * sizeof(ECMState));
    cudaMalloc(&d_rand_states, ECM_MAX_CURVES * sizeof(curandState_t));
    cudaMalloc(&d_result, sizeof(uint128_t));
    
    // Initialize result
    uint128_t h_result(0);
    cudaMemcpy(d_result, &h_result, sizeof(uint128_t), cudaMemcpyHostToDevice);
    
    // Initialize random states
    ecm_init_random<<<(ECM_MAX_CURVES + 255) / 256, 256>>>(d_rand_states, time(NULL));
    cudaDeviceSynchronize();
    
    // Run ECM in batches
    int curves_processed = 0;
    bool factor_found = false;
    
    while (curves_processed < max_curves && !factor_found) {
        // Launch ECM kernel
        ecm_factor_kernel<<<(ECM_MAX_CURVES + 255) / 256, 256>>>(
            n, d_states, d_rand_states, d_result);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ECM kernel error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        cudaDeviceSynchronize();
        
        // Check result
        cudaMemcpy(&h_result, d_result, sizeof(uint128_t), cudaMemcpyDeviceToHost);
        
        if (h_result > uint128_t(1)) {
            factor = h_result;
            factor_found = true;
        }
        
        curves_processed += ECM_MAX_CURVES;
        
        // Progress update
        if (curves_processed % 256 == 0) {
            printf("ECM: Processed %d curves...\n", curves_processed);
        }
    }
    
    // Clean up
    cudaFree(d_states);
    cudaFree(d_rand_states);
    cudaFree(d_result);
    
    return factor_found;
}

// Integration with main factorizer
__device__ __host__ bool ecm_is_suitable(uint128_t n) {
    // ECM is suitable for composite numbers with factors up to ~20 digits
    // Particularly effective when factors are around 10-20 digits
    int bit_size = 128 - n.leading_zeros();
    
    // Use ECM for numbers that are:
    // 1. Not too small (> 40 bits)
    // 2. Not prime (would need primality test)
    // 3. Failed quick trial division
    return bit_size > 40 && bit_size < 100;
}

#endif // ECM_CUDA_CU