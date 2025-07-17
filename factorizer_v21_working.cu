/**
 * Factorizer v2.1 Working - Pollard's Rho with correct arithmetic
 * Target: 15482526220500967432610341 (26 digits)
 * Known factors: 1804166129797 (0x1a410ae6885) × 8581541336353 (0x7ce0bb91521)
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

// Optimized modular multiplication using the fast method from uint128_improved.cuh
__device__ uint128_t modmul(const uint128_t& a, const uint128_t& b, const uint128_t& n) {
    return modmul_128_fast(a, b, n);
}

// Pollard's Rho function: x_{n+1} = x_n^2 + c (mod n)
__device__ uint128_t pollard_f(const uint128_t& x, const uint128_t& c, const uint128_t& n) {
    uint128_t x_squared = modmul(x, x, n);
    uint128_t result = add_128(x_squared, c);
    if (result >= n) {
        result = subtract_128(result, n);
    }
    return result;
}

// Verify factor by trial division (for small factors)
__global__ void verify_factor(uint128_t n, uint128_t factor) {
    printf("GPU: Verifying factor %llx:%llx\n", factor.high, factor.low);
    
    // Multiply factor by itself to check if it's close to n
    uint256_t product = multiply_128_128(factor, factor);
    printf("GPU: Factor squared = %llx:%llx:%llx:%llx\n",
           product.word[3], product.word[2], product.word[1], product.word[0]);
    
    // Check known factors
    uint128_t f1(0x1a410ae6885ULL, 0);  // 1804166129797
    uint128_t f2(0x7ce0bb91521ULL, 0);  // 8581541336353
    
    uint256_t known_product = multiply_128_128(f1, f2);
    printf("GPU: Known factors product = %llx:%llx:%llx:%llx\n",
           known_product.word[3], known_product.word[2], 
           known_product.word[1], known_product.word[0]);
    
    printf("GPU: Target n = %llx:%llx\n", n.high, n.low);
}

// Pollard's Rho kernel - optimized version
__global__ void pollard_rho_kernel(
    uint128_t n,
    uint128_t* device_factor,
    volatile int* device_status,
    int max_iterations,
    int thread_id_offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x + thread_id_offset;
    
    // Different starting values for each thread
    uint128_t x(2 + tid * 7, 0);
    uint128_t y = x;
    uint128_t c(1 + tid, 0);
    
    // Local variables
    int iterations = 0;
    const int batch_size = 100;  // Process in batches for efficiency
    
    while (iterations < max_iterations && *device_status == 0) {
        // Process a batch of iterations
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
            
            // Calculate GCD
            uint128_t d = gcd_128(diff, n);
            
            // Check if we found a non-trivial factor
            if (!d.is_zero() && d != uint128_t(1, 0) && d != n) {
                // Use atomic CAS to claim the discovery
                int expected = 0;
                if (atomicCAS((int*)device_status, expected, 1) == expected) {
                    *device_factor = d;
                    printf("GPU Thread %d: Found factor at iteration %d!\n", 
                           tid, iterations + i);
                    printf("GPU: Factor = %llx:%llx\n", d.high, d.low);
                }
                return;
            }
        }
        
        iterations += batch_size;
        
        // Progress report from thread 0
        if (tid == 0 && iterations % 10000 == 0) {
            printf("GPU: Thread 0 at iteration %d\n", iterations);
        }
        
        // Change parameters periodically
        if (iterations % 50000 == 0) {
            c = add_128(c, uint128_t(tid + 1, 0));
            x = uint128_t(2 + iterations + tid * 7, 0);
            y = x;
        }
    }
}

int main(int argc, char* argv[]) {
    // Target number: 15482526220500967432610341
    const char* target = "15482526220500967432610341";
    
    printf("CUDA Factorizer v2.1 Working\n");
    printf("Target: %s\n", target);
    printf("Known factors: 1804166129797 × 8581541336353\n\n");
    
    // Parse the target number
    uint128_t n = parse_decimal(target);
    printf("Parsed as uint128: high=0x%llx, low=0x%llx\n", n.high, n.low);
    
    // Verify parsing
    printf("Decimal representation: ");
    print_uint128_decimal(n);
    printf("\n\n");
    
    // First verify our arithmetic with known factors
    printf("Verifying arithmetic with known factors...\n");
    verify_factor<<<1, 1>>>(n, uint128_t(0, 0));
    cudaDeviceSynchronize();
    printf("\n");
    
    // Allocate device memory
    uint128_t* device_factor;
    int* device_status;
    cudaMalloc(&device_factor, sizeof(uint128_t));
    cudaMalloc(&device_status, sizeof(int));
    cudaMemset(device_status, 0, sizeof(int));
    
    // Run Pollard's Rho with multiple configurations
    const int num_blocks = 4;
    const int threads_per_block = 256;
    const int total_threads = num_blocks * threads_per_block;
    const int iterations_per_thread = 100000;
    
    printf("Running Pollard's Rho with %d threads...\n", total_threads);
    printf("Each thread will run up to %d iterations\n\n", iterations_per_thread);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    pollard_rho_kernel<<<num_blocks, threads_per_block>>>(
        n, device_factor, device_status, iterations_per_thread, 0
    );
    
    // Check periodically for result
    int status = 0;
    int check_count = 0;
    while (status == 0 && check_count < 60) {  // Check for up to 60 seconds
        sleep(1);
        cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
        check_count++;
        
        if (check_count % 5 == 0) {
            printf("Still searching... (%d seconds)\n", check_count);
        }
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Get final result
    cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
    uint128_t factor;
    cudaMemcpy(&factor, device_factor, sizeof(uint128_t), cudaMemcpyDeviceToHost);
    
    printf("\nComputation time: %.3f seconds\n", duration.count() / 1000.0);
    
    if (status == 1) {
        printf("\nFactor found: ");
        print_uint128_decimal(factor);
        printf("\n");
        printf("Hex: 0x%llx:%llx\n", factor.high, factor.low);
        
        // Check against known factors
        uint128_t f1(0x1a410ae6885ULL, 0);
        uint128_t f2(0x7ce0bb91521ULL, 0);
        
        if (factor == f1) {
            printf("This is the first known factor: 1804166129797\n");
            printf("The cofactor is: 8581541336353\n");
        } else if (factor == f2) {
            printf("This is the second known factor: 8581541336353\n");
            printf("The cofactor is: 1804166129797\n");
        } else {
            printf("This is a different factor!\n");
        }
    } else {
        printf("\nNo factor found in %d seconds.\n", check_count);
        printf("You may need to run longer or adjust parameters.\n");
    }
    
    // Cleanup
    cudaFree(device_factor);
    cudaFree(device_status);
    
    return 0;
}