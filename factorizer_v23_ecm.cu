/**
 * CUDA Factorizer v2.3.1 - ECM-focused Edition
 * Optimized for balanced factors using simplified ECM
 * 
 * Features:
 * - GPU-accelerated ECM implementation
 * - Optimized for 60-90 bit numbers with balanced factors
 * - Multiple parallel curves on GPU
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

// Include core components
#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"

// Version information
#define VERSION_MAJOR 2
#define VERSION_MINOR 3
#define VERSION_PATCH 1
#define VERSION_STRING "2.3.1-ECM"

// ECM Configuration
#define ECM_MAX_CURVES 1024       // Number of parallel curves per batch
#define ECM_STAGE1_BOUND 50000    // B1 bound for stage 1
#define ECM_THREADS_PER_BLOCK 256 // Threads per block

// Forward declarations
void print_uint128_decimal(uint128_t n);
uint128_t parse_decimal(const char* str);

// Simplified ECM point structure
struct ECMPoint {
    uint128_t x;
    uint128_t z;
};

// ECM curve parameters
struct ECMCurve {
    uint128_t a24;  // (A+2)/4 for Montgomery curves
    uint128_t n;
    Barrett128_v2 barrett;
};

// ECM state for each thread
struct ECMState {
    ECMPoint P;
    ECMPoint Q;
    ECMCurve curve;
    uint128_t factor;
    int found;
    curandState_t rand_state;
};

// Initialize random states
__global__ void init_ecm_random(curandState_t* states, int count, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Montgomery curve point doubling
__device__ void ecm_double(ECMPoint& R, const ECMPoint& P, const ECMCurve& curve) {
    // 2P = (X:Z) where:
    // X = (X+Z)^2 * (X-Z)^2
    // Z = 4XZ * ((X-Z)^2 + a24 * 4XZ)
    
    uint128_t xpz = add_128(P.x, P.z);
    if (xpz >= curve.n) xpz = subtract_128(xpz, curve.n);
    
    uint128_t xmz = (P.x >= P.z) ? subtract_128(P.x, P.z) : subtract_128(add_128(P.x, curve.n), P.z);
    
    uint256_t xpz2 = multiply_128_128(xpz, xpz);
    uint256_t xmz2 = multiply_128_128(xmz, xmz);
    
    uint128_t xpz2_red = curve.barrett.reduce(xpz2);
    uint128_t xmz2_red = curve.barrett.reduce(xmz2);
    
    // X coordinate
    uint256_t x_prod = multiply_128_128(xpz2_red, xmz2_red);
    R.x = curve.barrett.reduce(x_prod);
    
    // Z coordinate
    uint128_t diff = (xpz2_red >= xmz2_red) ? subtract_128(xpz2_red, xmz2_red) : 
                      subtract_128(add_128(xpz2_red, curve.n), xmz2_red);
    
    uint256_t a24_diff = multiply_128_128(curve.a24, diff);
    uint128_t a24_diff_red = curve.barrett.reduce(a24_diff);
    
    uint128_t sum = add_128(xmz2_red, a24_diff_red);
    if (sum >= curve.n) sum = subtract_128(sum, curve.n);
    
    uint256_t z_prod = multiply_128_128(diff, sum);
    R.z = curve.barrett.reduce(z_prod);
}

// Montgomery curve scalar multiplication using ladder
__device__ void ecm_scalar_mul(ECMPoint& R, const ECMPoint& P, uint64_t k, const ECMCurve& curve) {
    ECMPoint R0 = {uint128_t(1, 0), uint128_t(0, 0)};  // Point at infinity
    ECMPoint R1 = P;
    
    int bit_len = 64 - __clzll(k);
    
    for (int i = bit_len - 1; i >= 0; i--) {
        if ((k >> i) & 1) {
            ecm_double(R0, R0, curve);
            ecm_double(R1, R1, curve);
        } else {
            ecm_double(R1, R1, curve);
            ecm_double(R0, R0, curve);
        }
    }
    
    R = R0;
}

// ECM Stage 1 kernel
__global__ void ecm_stage1_kernel(ECMState* states, int count, uint128_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    ECMState& state = states[tid];
    
    // Initialize curve with n
    state.curve.n = n;
    state.curve.barrett.n = n;
    state.curve.barrett.precompute();
    
    // Generate random curve parameter
    uint64_t sigma = curand(&state.rand_state) % 1000000 + 2;
    
    // Simple curve generation: use sigma to create curve parameter
    state.curve.a24 = uint128_t(sigma, 0);
    
    // Starting point
    state.P.x = uint128_t(2, 0);
    state.P.z = uint128_t(1, 0);
    
    // Small primes for stage 1
    const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                         53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                         127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                         199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                         283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
                         383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                         467, 479, 487, 491, 499, 503, 509, 521, 523, 541};
    
    const int num_primes = sizeof(primes) / sizeof(primes[0]);
    
    // Accumulate point multiplications
    ECMPoint Q = state.P;
    
    for (int i = 0; i < num_primes && primes[i] <= ECM_STAGE1_BOUND; i++) {
        int p = primes[i];
        int pk = p;
        
        // Compute highest power of p <= B1
        while (pk <= ECM_STAGE1_BOUND / p) {
            pk *= p;
        }
        
        // Multiply by pk
        ecm_scalar_mul(Q, Q, pk, state.curve);
        
        // Check for factor periodically
        if (i % 20 == 19) {
            uint128_t g = gcd_128(Q.z, n);
            if (g > uint128_t(1, 0) && g < n) {
                state.factor = g;
                atomicExch(&state.found, 1);
                return;
            }
        }
    }
    
    // Final GCD check
    uint128_t g = gcd_128(Q.z, n);
    if (g > uint128_t(1, 0) && g < n) {
        state.factor = g;
        atomicExch(&state.found, 1);
    }
    
    state.Q = Q;
}

// Host function to run ECM
bool gpu_ecm_factor(uint128_t n, uint128_t& factor1, uint128_t& factor2, int max_curves = 10000) {
    printf("Starting GPU-accelerated ECM factorization...\n");
    printf("Target number: ");
    print_uint128_decimal(n);
    printf(" (%d bits)\n", 128 - n.leading_zeros());
    
    // Allocate device memory
    ECMState* d_states;
    curandState_t* d_rand_states;
    
    cudaMalloc(&d_states, ECM_MAX_CURVES * sizeof(ECMState));
    cudaMalloc(&d_rand_states, ECM_MAX_CURVES * sizeof(curandState_t));
    
    // Initialize random states
    int blocks = (ECM_MAX_CURVES + ECM_THREADS_PER_BLOCK - 1) / ECM_THREADS_PER_BLOCK;
    init_ecm_random<<<blocks, ECM_THREADS_PER_BLOCK>>>(d_rand_states, ECM_MAX_CURVES, time(NULL));
    cudaDeviceSynchronize();
    
    // Initialize states
    ECMState* h_states = new ECMState[ECM_MAX_CURVES];
    for (int i = 0; i < ECM_MAX_CURVES; i++) {
        h_states[i].found = 0;
        h_states[i].factor = uint128_t(0, 0);
    }
    cudaMemcpy(d_states, h_states, ECM_MAX_CURVES * sizeof(ECMState), cudaMemcpyHostToDevice);
    
    // Copy random states to ECM states
    for (int i = 0; i < ECM_MAX_CURVES; i++) {
        cudaMemcpy(&d_states[i].rand_state, &d_rand_states[i], sizeof(curandState_t), cudaMemcpyDeviceToDevice);
    }
    
    // Run ECM in batches
    int curves_done = 0;
    bool factor_found = false;
    uint128_t found_factor;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (curves_done < max_curves && !factor_found) {
        printf("Running ECM batch %d/%d (%d curves)...\n", 
               curves_done / ECM_MAX_CURVES + 1, 
               (max_curves + ECM_MAX_CURVES - 1) / ECM_MAX_CURVES,
               ECM_MAX_CURVES);
        
        // Run stage 1
        ecm_stage1_kernel<<<blocks, ECM_THREADS_PER_BLOCK>>>(d_states, ECM_MAX_CURVES, n);
        cudaDeviceSynchronize();
        
        // Check for factors
        cudaMemcpy(h_states, d_states, ECM_MAX_CURVES * sizeof(ECMState), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < ECM_MAX_CURVES; i++) {
            if (h_states[i].found) {
                found_factor = h_states[i].factor;
                factor_found = true;
                printf("Factor found by curve %d!\n", curves_done + i);
                break;
            }
        }
        
        curves_done += ECM_MAX_CURVES;
        
        // Progress report
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time).count();
        printf("Curves tested: %d, Time: %.1f seconds\n", curves_done, elapsed);
        
        // Re-randomize for next batch
        if (!factor_found && curves_done < max_curves) {
            init_ecm_random<<<blocks, ECM_THREADS_PER_BLOCK>>>(d_rand_states, ECM_MAX_CURVES, time(NULL) + curves_done);
            cudaDeviceSynchronize();
            
            // Reset states
            for (int i = 0; i < ECM_MAX_CURVES; i++) {
                h_states[i].found = 0;
                h_states[i].factor = uint128_t(0, 0);
            }
            cudaMemcpy(d_states, h_states, ECM_MAX_CURVES * sizeof(ECMState), cudaMemcpyHostToDevice);
            
            // Copy new random states
            for (int i = 0; i < ECM_MAX_CURVES; i++) {
                cudaMemcpy(&d_states[i].rand_state, &d_rand_states[i], sizeof(curandState_t), cudaMemcpyDeviceToDevice);
            }
        }
    }
    
    // Clean up
    delete[] h_states;
    cudaFree(d_states);
    cudaFree(d_rand_states);
    
    if (factor_found) {
        factor1 = found_factor;
        
        // Calculate factor2
        if (n.high == 0 && found_factor.high == 0) {
            factor2 = uint128_t(n.low / found_factor.low, 0);
        } else {
            // Use Barrett division
            uint256_t n_256;
            n_256.word[0] = n.low;
            n_256.word[1] = n.high;
            n_256.word[2] = 0;
            n_256.word[3] = 0;
            factor2 = divide_256_128(n_256, found_factor);
        }
        
        // Verify
        uint256_t check = multiply_128_128(factor1, factor2);
        if (uint128_t(check.word[0], check.word[1]) == n) {
            return true;
        }
    }
    
    return false;
}

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
        printf("%llu", (unsigned long long)n.low);
        return;
    }
    
    char buffer[40];
    int pos = 39;
    buffer[pos] = '\0';
    
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

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        printf("Example: %s 139789207152250802634791\n", argv[0]);
        return 1;
    }
    
    // Parse number
    uint128_t n = parse_decimal(argv[1]);
    
    // Check CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("Error: No CUDA-capable devices found\n");
        return 1;
    }
    
    cudaSetDevice(0);
    
    printf("CUDA Factorizer v%s - ECM-focused Edition\n", VERSION_STRING);
    printf("GPU-accelerated ECM for balanced factors\n\n");
    
    // Run ECM
    uint128_t factor1, factor2;
    bool success = gpu_ecm_factor(n, factor1, factor2, 50000);
    
    if (success) {
        printf("\n✓ Factorization successful!\n");
        printf("Factor 1: ");
        print_uint128_decimal(factor1);
        printf("\nFactor 2: ");
        print_uint128_decimal(factor2);
        printf("\n");
        
        // Verify
        uint256_t check = multiply_128_128(factor1, factor2);
        if (uint128_t(check.word[0], check.word[1]) == n) {
            printf("✓ Verification: factors multiply to original number\n");
        }
        
        return 0;
    } else {
        printf("\n✗ Factorization failed\n");
        return 1;
    }
}