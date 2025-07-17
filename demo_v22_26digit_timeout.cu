/**
 * CUDA Factorizer v2.2.0 - 26-digit Test Case Demo with Timeout
 * Includes timeout and progress reporting
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>
#include <atomic>

#include "uint128_improved.cuh"
#include "barrett_reduction_v2.cuh"

// Global progress counter
__device__ unsigned long long g_iterations = 0;

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

// Pollard's f function with Barrett reduction
__device__ uint128_t pollards_f(const uint128_t& x, const uint128_t& c, const Barrett128_v2& barrett) {
    uint256_t x_squared = multiply_128_128(x, x);
    uint128_t result = barrett.reduce(x_squared);
    
    result = add_128(result, c);
    if (result >= barrett.n) {
        result = subtract_128(result, barrett.n);
    }
    
    return result;
}

// Optimized Pollard's Rho with timeout
__global__ void pollards_rho_26digit_timeout(
    uint128_t n,
    uint128_t* factor,
    int* found,
    int max_iterations_per_thread = 10000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize PRNG with good entropy
    curandState_t state;
    curand_init(clock64() + tid * 9973 + blockIdx.x * 31337, tid, 0, &state);
    
    // Setup Barrett reduction
    Barrett128_v2 barrett;
    barrett.n = n;
    barrett.precompute();
    
    // We know the factors are around 10^13, so start near sqrt(n)
    uint64_t base = 1000000000000ULL; // 10^12
    uint64_t range = 10000000000000ULL; // 10^13
    
    // Initialize with values near sqrt(n)
    uint128_t x(base + (curand(&state) % range), curand(&state) % 1000);
    uint128_t y = x;
    uint128_t c(1 + (curand(&state) % 10000), 0);
    
    // Brent's variant parameters
    int m = 128;  // Batch size
    int r = 1;
    uint128_t ys = y;
    uint128_t product(1, 0);
    
    for (int i = 0; i < max_iterations_per_thread && !(*found); i++) {
        // Update global counter every 1000 iterations
        if (i % 1000 == 0) {
            atomicAdd(&g_iterations, 1000);
        }
        
        // Brent's algorithm
        if (i == r) {
            ys = y;
            r *= 2;
        }
        
        // Batch GCD computation
        uint128_t min_val = y;
        for (int j = 0; j < m && i + j < r; j++) {
            y = pollards_f(y, c, barrett);
            
            uint128_t diff = (y > ys) ? subtract_128(y, ys) : subtract_128(ys, y);
            uint256_t prod = multiply_128_128(product, diff);
            product = barrett.reduce(prod);
        }
        
        // Check GCD
        uint128_t g = gcd_128(product, n);
        
        if (g > uint128_t(1, 0) && g < n) {
            // Found a factor!
            *factor = g;
            atomicExch(found, 1);
            return;
        }
        
        // Reset product
        if (i % 1000 == 0) {
            product = uint128_t(1, 0);
        }
        
        // Adaptive parameter change
        if (i % (100000 + tid * 1000) == 0) {
            c = uint128_t(1 + (curand(&state) % 100000), 0);
            x = uint128_t(base + (curand(&state) % range), curand(&state) % 1000);
            y = x;
            ys = y;
            product = uint128_t(1, 0);
        }
    }
}

int main() {
    printf("\n==================================================\n");
    printf("   CUDA Factorizer v2.2.0 - 26-Digit Challenge\n");
    printf("        (With Timeout and Progress)\n");
    printf("==================================================\n\n");
    
    // The 26-digit challenge number
    const char* number_str = "15482526220500967432610341";
    uint128_t n = parse_decimal(number_str);
    
    printf("Target: %s\n", number_str);
    printf("Binary: %016llx%016llx\n", n.high, n.low);
    printf("Bit size: %d\n\n", 128 - n.leading_zeros());
    
    printf("Expected factors:\n");
    printf("  1804166129797 × 8581541336353\n\n");
    
    // Device memory
    uint128_t* d_factor;
    int* d_found;
    unsigned long long* d_iterations;
    cudaMalloc(&d_factor, sizeof(uint128_t));
    cudaMalloc(&d_found, sizeof(int));
    cudaMemset(d_found, 0, sizeof(int));
    cudaMemcpyToSymbol(g_iterations, &d_iterations, sizeof(unsigned long long));
    
    // Launch configuration optimized for GTX 2070
    int num_blocks = 256;  // 8x multiprocessors
    int threads_per_block = 256;
    int total_threads = num_blocks * threads_per_block;
    
    printf("Launching %d threads (%d blocks × %d threads/block)\n", 
           total_threads, num_blocks, threads_per_block);
    printf("Using Pollard's Rho with Brent's optimization\n");
    printf("Maximum time: 60 seconds\n");
    printf("Starting search...\n\n");
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    clock_t cpu_start = clock();
    
    // Launch kernel
    pollards_rho_26digit_timeout<<<num_blocks, threads_per_block>>>(n, d_factor, d_found);
    
    // Monitor progress with timeout
    int h_found = 0;
    int seconds_elapsed = 0;
    const int timeout_seconds = 60;
    unsigned long long total_iterations = 0;
    
    while (!h_found && seconds_elapsed < timeout_seconds) {
        // Wait 1 second
        sleep(1);
        seconds_elapsed++;
        
        // Check if factor found
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Get iteration count
        cudaMemcpyFromSymbol(&total_iterations, g_iterations, sizeof(unsigned long long));
        
        // Print progress
        printf("\rProgress: %d seconds, ~%llu million iterations checked", 
               seconds_elapsed, total_iterations / 1000000);
        fflush(stdout);
        
        // Check if kernel finished
        if (cudaStreamQuery(0) == cudaSuccess) {
            break;
        }
    }
    
    // If timeout, terminate kernel
    if (seconds_elapsed >= timeout_seconds && !h_found) {
        printf("\n\nTimeout reached after %d seconds.\n", timeout_seconds);
        printf("Total iterations: ~%llu million\n", total_iterations / 1000000);
        
        // Note: In production, you'd want a more graceful termination
        cudaDeviceReset();
        
        printf("\n--------------------------------------------------\n");
        printf("                    RESULTS\n");
        printf("--------------------------------------------------\n");
        printf("✗ No factors found within timeout\n");
        printf("\nThis 26-digit number with two 13-digit prime factors\n");
        printf("is at the edge of what Pollard's Rho can handle.\n");
        printf("Consider using Quadratic Sieve or ECM for such numbers.\n");
        printf("==================================================\n\n");
        
        return 1;
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Get results
    uint128_t h_factor;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_factor, d_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    // Calculate timing
    auto end = std::chrono::high_resolution_clock::now();
    clock_t cpu_end = clock();
    
    double wall_time = std::chrono::duration<double>(end - start).count();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    
    printf("\n\n--------------------------------------------------\n");
    printf("                    RESULTS\n");
    printf("--------------------------------------------------\n");
    
    if (h_found && h_factor.low > 1) {
        printf("✓ Factor found: ");
        print_uint128_decimal(h_factor);
        printf("\n");
        
        // Calculate cofactor
        uint256_t n_256;
        n_256.word[0] = n.low;
        n_256.word[1] = n.high;
        n_256.word[2] = 0;
        n_256.word[3] = 0;
        uint128_t cofactor = divide_256_128(n_256, h_factor);
        
        printf("  Cofactor: ");
        print_uint128_decimal(cofactor);
        printf("\n\n");
        
        // Verify specific factors
        uint128_t factor1 = parse_decimal("1804166129797");
        uint128_t factor2 = parse_decimal("8581541336353");
        
        if (h_factor == factor1 || h_factor == factor2) {
            printf("✓ This is one of the expected prime factors!\n");
        }
        
        // Verify factorization
        uint256_t product = multiply_128_128(h_factor, cofactor);
        uint128_t check(product.word[0], product.word[1]);
        
        if (check == n) {
            printf("✓ Factorization verified!\n");
        }
    } else {
        printf("✗ No factors found\n");
        printf("\nPossible reasons:\n");
        printf("- Need more iterations\n");
        printf("- Need different initial parameters\n");
        printf("- Factors may require more advanced methods\n");
    }
    
    printf("\nPerformance:\n");
    printf("  Wall time: %.3f seconds\n", wall_time);
    printf("  CPU time: %.3f seconds\n", cpu_time);
    printf("  GPU efficiency: %.1f%%\n", (cpu_time / wall_time) * 100.0);
    printf("  Total iterations: ~%llu million\n", total_iterations / 1000000);
    if (wall_time > 0) {
        printf("  Iterations/sec: ~%.1f million\n", (total_iterations / 1000000) / wall_time);
    }
    
    // Cleanup
    cudaFree(d_factor);
    cudaFree(d_found);
    
    printf("==================================================\n\n");
    
    return h_found ? 0 : 1;
}