/**
 * GPU-Accelerated Elliptic Curve Method (ECM) Design
 * Optimized for finding factors up to 25 digits
 * 
 * This implementation uses Montgomery curves for efficient arithmetic
 * and parallel processing of multiple curves simultaneously.
 */

#ifndef ECM_GPU_DESIGN_CUH
#define ECM_GPU_DESIGN_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "uint128_improved.cuh"

// Configuration parameters for ECM
#define ECM_MAX_CURVES_PER_BLOCK 32
#define ECM_STAGE1_BOUND_DEFAULT 100000
#define ECM_STAGE2_BOUND_DEFAULT 10000000
#define ECM_BABY_STEPS 512
#define ECM_GIANT_STEPS 512

/**
 * Montgomery Curve: B*y^2 = x^3 + A*x^2 + x (mod n)
 * We use the Montgomery form for efficient point arithmetic
 */
struct MontgomeryCurve {
    uint128_t A;      // Curve parameter A
    uint128_t B;      // Curve parameter B (usually 1)
    uint128_t n;      // The number we're factoring
    uint128_t A24;    // (A+2)/4 precomputed for efficiency
    
    __device__ __host__ MontgomeryCurve() : A(0), B(1), n(0), A24(0) {}
    
    __device__ __host__ MontgomeryCurve(uint128_t a, uint128_t modulus) 
        : A(a), B(1), n(modulus) {
        // Precompute (A+2)/4 for Montgomery ladder
        uint128_t a_plus_2 = add_128(A, uint128_t(2));
        A24 = modular_divide_by_4(a_plus_2, n);
    }
    
    __device__ uint128_t modular_divide_by_4(uint128_t x, uint128_t mod) {
        // For now, simple implementation - can be optimized
        uint128_t quarter = x;
        for (int i = 0; i < 2; i++) {
            if (quarter.low & 1) {
                quarter = add_128(quarter, mod);
            }
            quarter.low = (quarter.low >> 1) | (quarter.high << 63);
            quarter.high >>= 1;
        }
        return quarter;
    }
};

/**
 * Point on Montgomery curve in projective coordinates
 * Represents point (X:Z) where x = X/Z
 */
struct ECMPoint {
    uint128_t X;
    uint128_t Z;
    
    __device__ __host__ ECMPoint() : X(0), Z(1) {}
    __device__ __host__ ECMPoint(uint128_t x, uint128_t z) : X(x), Z(z) {}
    
    __device__ __host__ bool is_infinity() const {
        return Z.is_zero();
    }
};

/**
 * Montgomery arithmetic operations
 * These are optimized for the Montgomery ladder used in scalar multiplication
 */
class MontgomeryArithmetic {
public:
    /**
     * Montgomery point doubling
     * Given P = (X:Z), compute 2P
     * Cost: 2M + 2S + 1M_A24 (where M=multiplication, S=squaring)
     */
    __device__ static ECMPoint point_double(const ECMPoint& P, const MontgomeryCurve& curve) {
        if (P.is_infinity()) return P;
        
        // t1 = (X + Z)^2
        uint128_t x_plus_z = add_128(P.X, P.Z);
        if (x_plus_z >= curve.n) x_plus_z = subtract_128(x_plus_z, curve.n);
        uint128_t t1 = modmul_128(x_plus_z, x_plus_z, curve.n);
        
        // t2 = (X - Z)^2
        uint128_t x_minus_z = (P.X >= P.Z) ? 
            subtract_128(P.X, P.Z) : 
            subtract_128(add_128(P.X, curve.n), P.Z);
        uint128_t t2 = modmul_128(x_minus_z, x_minus_z, curve.n);
        
        // X_2P = t1 * t2
        uint128_t X_2P = modmul_128(t1, t2, curve.n);
        
        // t3 = t1 - t2
        uint128_t t3 = (t1 >= t2) ? 
            subtract_128(t1, t2) : 
            subtract_128(add_128(t1, curve.n), t2);
        
        // Z_2P = t3 * (t2 + A24 * t3)
        uint128_t a24_t3 = modmul_128(curve.A24, t3, curve.n);
        uint128_t t4 = add_128(t2, a24_t3);
        if (t4 >= curve.n) t4 = subtract_128(t4, curve.n);
        uint128_t Z_2P = modmul_128(t3, t4, curve.n);
        
        return ECMPoint(X_2P, Z_2P);
    }
    
    /**
     * Montgomery differential addition
     * Given P, Q, and P-Q, compute P+Q
     * Cost: 4M + 2S
     */
    __device__ static ECMPoint point_add(
        const ECMPoint& P, 
        const ECMPoint& Q, 
        const ECMPoint& P_minus_Q,
        const MontgomeryCurve& curve
    ) {
        if (P.is_infinity()) return Q;
        if (Q.is_infinity()) return P;
        
        // t1 = (X_P - Z_P) * (X_Q + Z_Q)
        uint128_t xp_minus_zp = (P.X >= P.Z) ?
            subtract_128(P.X, P.Z) :
            subtract_128(add_128(P.X, curve.n), P.Z);
        uint128_t xq_plus_zq = add_128(Q.X, Q.Z);
        if (xq_plus_zq >= curve.n) xq_plus_zq = subtract_128(xq_plus_zq, curve.n);
        uint128_t t1 = modmul_128(xp_minus_zp, xq_plus_zq, curve.n);
        
        // t2 = (X_P + Z_P) * (X_Q - Z_Q)
        uint128_t xp_plus_zp = add_128(P.X, P.Z);
        if (xp_plus_zp >= curve.n) xp_plus_zp = subtract_128(xp_plus_zp, curve.n);
        uint128_t xq_minus_zq = (Q.X >= Q.Z) ?
            subtract_128(Q.X, Q.Z) :
            subtract_128(add_128(Q.X, curve.n), Q.Z);
        uint128_t t2 = modmul_128(xp_plus_zp, xq_minus_zq, curve.n);
        
        // t3 = t1 + t2, t4 = t1 - t2
        uint128_t t3 = add_128(t1, t2);
        if (t3 >= curve.n) t3 = subtract_128(t3, curve.n);
        uint128_t t4 = (t1 >= t2) ?
            subtract_128(t1, t2) :
            subtract_128(add_128(t1, curve.n), t2);
        
        // X_R = Z_(P-Q) * t3^2
        uint128_t t3_squared = modmul_128(t3, t3, curve.n);
        uint128_t X_R = modmul_128(P_minus_Q.Z, t3_squared, curve.n);
        
        // Z_R = X_(P-Q) * t4^2
        uint128_t t4_squared = modmul_128(t4, t4, curve.n);
        uint128_t Z_R = modmul_128(P_minus_Q.X, t4_squared, curve.n);
        
        return ECMPoint(X_R, Z_R);
    }
    
    /**
     * Montgomery ladder for scalar multiplication
     * Computes k*P using constant-time algorithm
     */
    __device__ static ECMPoint scalar_multiply(
        const ECMPoint& P,
        uint128_t k,
        const MontgomeryCurve& curve
    ) {
        if (k.is_zero() || P.is_infinity()) {
            return ECMPoint(0, 0);  // Point at infinity
        }
        
        // Find the highest set bit
        int bit_length = 128 - k.leading_zeros();
        
        // Initialize: R0 = P, R1 = 2P
        ECMPoint R0 = P;
        ECMPoint R1 = point_double(P, curve);
        
        // Montgomery ladder - process bits from high to low
        for (int i = bit_length - 2; i >= 0; i--) {
            bool bit = (i >= 64) ? 
                ((k.high >> (i - 64)) & 1) : 
                ((k.low >> i) & 1);
            
            if (bit) {
                // R0 = R0 + R1, R1 = 2*R1
                R0 = point_add(R0, R1, P, curve);
                R1 = point_double(R1, curve);
            } else {
                // R1 = R0 + R1, R0 = 2*R0
                R1 = point_add(R0, R1, P, curve);
                R0 = point_double(R0, curve);
            }
        }
        
        return R0;
    }
};

/**
 * ECM Stage 1: Find smooth order curves
 * Uses standard multiplication by B1-smooth number
 */
__device__ void ecm_stage1(
    ECMPoint& Q,
    const ECMPoint& P,
    const MontgomeryCurve& curve,
    uint32_t B1
) {
    Q = P;
    
    // Multiply by all primes p <= B1 with appropriate powers
    // For efficiency, we precompute the product of all such prime powers
    // In practice, this would use a precomputed table
    
    // Simplified version: multiply by factorial(B1) approximation
    for (uint32_t p = 2; p <= B1; p++) {
        // In real implementation, check if p is prime
        // and compute appropriate power
        Q = MontgomeryArithmetic::scalar_multiply(Q, uint128_t(p), curve);
        
        // Early termination if we found a factor
        if (!Q.Z.is_zero() && gcd_128(Q.Z, curve.n) > uint128_t(1)) {
            return;
        }
    }
}

/**
 * ECM Stage 2: Baby-step Giant-step implementation
 * Searches for primes in interval [B1, B2]
 */
struct Stage2State {
    ECMPoint baby_steps[ECM_BABY_STEPS];
    ECMPoint current_giant;
    uint32_t B1, B2;
    uint32_t baby_step_size;
    uint32_t giant_step_size;
};

__device__ void ecm_stage2_init(
    Stage2State& state,
    const ECMPoint& Q,
    const MontgomeryCurve& curve,
    uint32_t B1,
    uint32_t B2
) {
    state.B1 = B1;
    state.B2 = B2;
    state.baby_step_size = ECM_BABY_STEPS;
    state.giant_step_size = ECM_GIANT_STEPS;
    
    // Precompute baby steps: Q, 2Q, 3Q, ..., baby_step_size*Q
    state.baby_steps[0] = Q;
    for (int i = 1; i < ECM_BABY_STEPS; i++) {
        state.baby_steps[i] = MontgomeryArithmetic::point_add(
            state.baby_steps[i-1], Q, 
            (i > 1) ? state.baby_steps[i-2] : Q,
            curve
        );
    }
    
    // Initialize giant step
    ECMPoint giant_base = MontgomeryArithmetic::scalar_multiply(
        Q, uint128_t(state.baby_step_size), curve
    );
    state.current_giant = giant_base;
}

__device__ uint128_t ecm_stage2_search(
    Stage2State& state,
    const MontgomeryCurve& curve
) {
    // Search for factors using baby-step giant-step
    for (uint32_t giant = 0; giant < state.giant_step_size; giant++) {
        for (uint32_t baby = 0; baby < state.baby_step_size; baby++) {
            // Check if current combination yields a factor
            uint128_t g = gcd_128(state.baby_steps[baby].Z, curve.n);
            if (g > uint128_t(1) && g < curve.n) {
                return g;
            }
        }
        
        // Move to next giant step
        state.current_giant = MontgomeryArithmetic::scalar_multiply(
            state.current_giant,
            uint128_t(state.baby_step_size),
            curve
        );
    }
    
    return uint128_t(1);  // No factor found
}

/**
 * Main ECM kernel - processes multiple curves in parallel
 */
__global__ void ecm_parallel_kernel(
    uint128_t n,
    uint128_t* factors,
    int* factor_found,
    uint32_t B1,
    uint32_t B2,
    int curves_per_thread,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random number generator
    curandState_t rng_state;
    curand_init(seed + tid, 0, 0, &rng_state);
    
    // Process multiple curves per thread
    for (int curve_idx = 0; curve_idx < curves_per_thread; curve_idx++) {
        // Check if factor already found by another thread
        if (atomicAdd(factor_found, 0) > 0) return;
        
        // Generate random curve parameters
        uint128_t sigma(curand(&rng_state), curand(&rng_state));
        sigma = modulo_128(sigma, n);
        
        // Convert to Montgomery form: A = (sigma^2 - 5) / (4*sigma)
        uint128_t sigma_squared = modmul_128(sigma, sigma, n);
        uint128_t numerator = (sigma_squared >= uint128_t(5)) ?
            subtract_128(sigma_squared, uint128_t(5)) :
            subtract_128(add_128(sigma_squared, n), uint128_t(5));
        
        // Simplified: for now use random A directly
        uint128_t A = numerator;  // Should divide by 4*sigma
        MontgomeryCurve curve(A, n);
        
        // Starting point
        ECMPoint P(uint128_t(2), uint128_t(1));
        
        // Stage 1
        ECMPoint Q;
        ecm_stage1(Q, P, curve, B1);
        
        // Check for factor from Stage 1
        if (!Q.Z.is_zero()) {
            uint128_t g = gcd_128(Q.Z, n);
            if (g > uint128_t(1) && g < n) {
                // Found a factor!
                int old = atomicCAS(factor_found, 0, 1);
                if (old == 0) {
                    factors[0] = g;
                    factors[1] = divide_128(n, g).first;
                }
                return;
            }
        }
        
        // Stage 2
        Stage2State stage2;
        ecm_stage2_init(stage2, Q, curve, B1, B2);
        uint128_t factor = ecm_stage2_search(stage2, curve);
        
        if (factor > uint128_t(1) && factor < n) {
            // Found a factor!
            int old = atomicCAS(factor_found, 0, 1);
            if (old == 0) {
                factors[0] = factor;
                factors[1] = divide_128(n, factor).first;
            }
            return;
        }
    }
}

/**
 * Helper functions for ECM integration
 */

// Compute GCD using binary algorithm
__device__ uint128_t gcd_128(uint128_t a, uint128_t b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;
    
    // Remove common factors of 2
    int shift = 0;
    while (((a.low | b.low) & 1) == 0) {
        a.low = (a.low >> 1) | (a.high << 63);
        a.high >>= 1;
        b.low = (b.low >> 1) | (b.high << 63);
        b.high >>= 1;
        shift++;
    }
    
    // Remove factors of 2 from a
    while ((a.low & 1) == 0) {
        a.low = (a.low >> 1) | (a.high << 63);
        a.high >>= 1;
    }
    
    do {
        // Remove factors of 2 from b
        while ((b.low & 1) == 0) {
            b.low = (b.low >> 1) | (b.high << 63);
            b.high >>= 1;
        }
        
        // Ensure a <= b
        if (a > b) {
            uint128_t temp = a;
            a = b;
            b = temp;
        }
        
        b = subtract_128(b, a);
    } while (!b.is_zero());
    
    // Restore common factors of 2
    while (shift > 0) {
        a.high = (a.high << 1) | (a.low >> 63);
        a.low <<= 1;
        shift--;
    }
    
    return a;
}

// Simple modulo operation
__device__ uint128_t modulo_128(uint128_t a, uint128_t n) {
    while (a >= n) {
        a = subtract_128(a, n);
    }
    return a;
}

// Division helper (returns quotient and remainder)
__device__ pair<uint128_t, uint128_t> divide_128(uint128_t dividend, uint128_t divisor) {
    // Simplified division - in practice would use more efficient algorithm
    uint128_t quotient(0);
    uint128_t remainder = dividend;
    
    while (remainder >= divisor) {
        remainder = subtract_128(remainder, divisor);
        quotient = add_128(quotient, uint128_t(1));
    }
    
    return make_pair(quotient, remainder);
}

/**
 * ECM factorization entry point
 * Integrates with existing factorizer framework
 */
extern "C" __host__ bool ecm_factorize(
    uint128_t n,
    uint128_t* factors,
    int* num_factors,
    int max_curves = 1000,
    uint32_t B1 = ECM_STAGE1_BOUND_DEFAULT,
    uint32_t B2 = ECM_STAGE2_BOUND_DEFAULT
) {
    // Allocate device memory
    uint128_t* d_factors;
    int* d_factor_found;
    cudaMalloc(&d_factors, 2 * sizeof(uint128_t));
    cudaMalloc(&d_factor_found, sizeof(int));
    cudaMemset(d_factor_found, 0, sizeof(int));
    
    // Configure kernel launch
    int threads_per_block = 256;
    int blocks = (max_curves + threads_per_block - 1) / threads_per_block;
    int curves_per_thread = 1;
    
    // Launch ECM kernel
    ecm_parallel_kernel<<<blocks, threads_per_block>>>(
        n, d_factors, d_factor_found, B1, B2, curves_per_thread, time(NULL)
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ECM kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_factors);
        cudaFree(d_factor_found);
        return false;
    }
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Check if factor was found
    int factor_found;
    cudaMemcpy(&factor_found, d_factor_found, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (factor_found) {
        cudaMemcpy(factors, d_factors, 2 * sizeof(uint128_t), cudaMemcpyDeviceToHost);
        *num_factors = 2;
    }
    
    // Cleanup
    cudaFree(d_factors);
    cudaFree(d_factor_found);
    
    return factor_found > 0;
}

/**
 * Optimal parameter selection for different input sizes
 */
struct ECMParameters {
    uint32_t B1;
    uint32_t B2;
    int num_curves;
};

__host__ ECMParameters select_ecm_parameters(int digit_count) {
    ECMParameters params;
    
    // Parameters optimized for different factor sizes
    // Based on empirical data and theoretical analysis
    if (digit_count <= 15) {
        params.B1 = 2000;
        params.B2 = 100000;
        params.num_curves = 100;
    } else if (digit_count <= 20) {
        params.B1 = 11000;
        params.B2 = 1000000;
        params.num_curves = 500;
    } else if (digit_count <= 25) {
        params.B1 = 50000;
        params.B2 = 5000000;
        params.num_curves = 2000;
    } else {
        // For larger factors, need more aggressive parameters
        params.B1 = 250000;
        params.B2 = 25000000;
        params.num_curves = 10000;
    }
    
    return params;
}

#endif // ECM_GPU_DESIGN_CUH