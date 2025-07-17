/**
 * Factorizer v2.1 Clean - Pollard's Rho with efficient modular arithmetic
 * Target: 15482526220500967432610341 (26 digits)
 * Known factors: 1804166129797 × 8581541336353
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <unistd.h>
#include "uint128_improved.cuh"

// Parse decimal string to uint128
uint128_t parse_decimal(const char* str) {
    uint128_t result(0, 0);
    uint128_t ten(10, 0);
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= '0' && str[i] <= '9') {
            // result = result * 10 + digit
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
    
    // Convert to string by repeated division
    char buffer[40];
    int pos = 39;
    buffer[pos] = '\0';
    
    while (!n.is_zero() && pos > 0) {
        // Division by 10 using simple method
        uint128_t quotient(0, 0);
        uint64_t remainder = 0;
        
        // Process high word
        if (n.high > 0) {
            remainder = n.high % 10;
            quotient.high = n.high / 10;
        }
        
        // Process low word with carry
        unsigned __int128 temp = (unsigned __int128)remainder << 64 | n.low;
        quotient.low = temp / 10;
        remainder = temp % 10;
        
        buffer[--pos] = '0' + remainder;
        n = quotient;
    }
    
    printf("%s", &buffer[pos]);
}

// Pollard's Rho function: x_{n+1} = x_n^2 + c (mod n)
__device__ uint128_t pollard_f(const uint128_t& x, const uint128_t& c, const uint128_t& n) {
    uint128_t x_squared = modmul_128_fast(x, x, n);
    uint128_t result = add_128(x_squared, c);
    if (result >= n) {
        result = subtract_128(result, n);
    }
    return result;
}

// Main Pollard's Rho kernel
__global__ void pollard_rho_clean(
    uint128_t n,
    uint128_t* device_factor,
    volatile int* device_status,
    int max_iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random number generator
    curandState state;
    curand_init(clock64() + tid * 997, tid, 0, &state);
    
    // Starting values - use better seeds
    uint64_t seed = curand(&state);
    uint128_t x(seed % (1ULL << 32), 0);
    uint128_t y = x;
    uint128_t c((seed >> 32) % 1000 + 1, 0);
    
    int iterations = 0;
    const int batch_size = 128;
    
    while (iterations < max_iterations && *device_status == 0) {
        // Process a batch of iterations
        uint128_t batch_product(1, 0);
        int batch_count = 0;
        
        for (int i = 0; i < batch_size && *device_status == 0; i++) {
            // Floyd's cycle detection
            x = pollard_f(x, c, n);
            y = pollard_f(pollard_f(y, c, n), c, n);
            
            // Calculate |x - y|
            uint128_t diff;
            if (x >= y) {
                diff = subtract_128(x, y);
            } else {
                diff = subtract_128(y, x);
            }
            
            // Skip if diff is 0
            if (diff.is_zero()) continue;
            
            // Accumulate differences (Montgomery's trick)
            batch_product = modmul_128_fast(batch_product, diff, n);
            batch_count++;
            
            // Check batch GCD periodically
            if (batch_count >= 20 || batch_product.is_zero()) {
                uint128_t d = gcd_128(batch_product, n);
                
                if (!d.is_zero() && d != uint128_t(1, 0) && d != n) {
                    // Found a factor!
                    int expected = 0;
                    if (atomicCAS((int*)device_status, expected, 1) == expected) {
                        *device_factor = d;
                        printf("Thread %d: Found factor after %d iterations\n", 
                               tid, iterations + i);
                    }
                    return;
                }
                
                // Reset batch
                batch_product = uint128_t(1, 0);
                batch_count = 0;
            }
        }
        
        iterations += batch_size;
        
        // Progress from thread 0
        if (tid == 0 && iterations % 100000 == 0) {
            printf("Progress: %d iterations\n", iterations);
        }
        
        // Change parameters periodically
        if (iterations % 500000 == 0) {
            seed = curand(&state);
            x = uint128_t(seed % (1ULL << 32), 0);
            y = x;
            c = uint128_t((seed >> 32) % 1000 + 1, 0);
            if (tid == 0) {
                printf("Thread 0: Restarting with new parameters\n");
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Target number: 15482526220500967432610341
    const char* target = "15482526220500967432610341";
    
    printf("CUDA Factorizer v2.1 Clean\n");
    printf("Target: %s\n", target);
    printf("Known factors: 1804166129797 × 8581541336353\n\n");
    
    // Parse the target number
    uint128_t n = parse_decimal(target);
    printf("Parsed: ");
    print_uint128_decimal(n);
    printf(" (0x%llx:%llx)\n\n", n.high, n.low);
    
    // Allocate device memory
    uint128_t* device_factor;
    int* device_status;
    cudaMalloc(&device_factor, sizeof(uint128_t));
    cudaMalloc(&device_status, sizeof(int));
    cudaMemset(device_status, 0, sizeof(int));
    
    // Configuration
    const int num_blocks = 16;
    const int threads_per_block = 256;
    const int total_threads = num_blocks * threads_per_block;
    const int iterations_per_thread = 2000000;
    
    printf("Running Pollard's Rho with %d threads\n", total_threads);
    printf("Maximum iterations per thread: %d\n\n", iterations_per_thread);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    pollard_rho_clean<<<num_blocks, threads_per_block>>>(
        n, device_factor, device_status, iterations_per_thread
    );
    
    // Monitor progress
    int status = 0;
    int seconds = 0;
    
    while (status == 0 && seconds < 300) {  // 5 minute timeout
        sleep(1);
        seconds++;
        
        cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (seconds % 10 == 0 && status == 0) {
            printf("Elapsed: %d seconds\n", seconds);
        }
    }
    
    // Ensure kernel completes
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Get result
    cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
    uint128_t factor;
    cudaMemcpy(&factor, device_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    printf("\nTotal time: %.3f seconds\n", duration.count() / 1000.0);
    
    if (status == 1) {
        printf("\n=== FACTOR FOUND ===\n");
        printf("Factor: ");
        print_uint128_decimal(factor);
        printf(" (0x%llx:%llx)\n", factor.high, factor.low);
        
        // Calculate cofactor using simple division
        printf("\nVerifying factorization...\n");
        printf("%s = ", target);
        print_uint128_decimal(factor);
        printf(" × [cofactor]\n");
        
        // Check against known factors
        uint128_t f1(0x1a410ae6885ULL, 0);  // 1804166129797
        uint128_t f2(0x7ce0bb91521ULL, 0);  // 8581541336353
        
        if (factor == f1) {
            printf("\nThis matches the first known factor!\n");
            printf("Cofactor should be: 8581541336353\n");
        } else if (factor == f2) {
            printf("\nThis matches the second known factor!\n");
            printf("Cofactor should be: 1804166129797\n");
        }
    } else {
        printf("\nNo factor found within time limit.\n");
        printf("The algorithm may need more iterations or different parameters.\n");
    }
    
    // Cleanup
    cudaFree(device_factor);
    cudaFree(device_status);
    
    return 0;
}