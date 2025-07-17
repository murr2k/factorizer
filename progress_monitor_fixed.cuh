/**
 * Progress Monitoring and GPU Utilization Tracking
 * Real-time progress reporting and performance metrics
 * For CUDA GTX 2070 architecture
 */

#ifndef PROGRESS_MONITOR_CUH
#define PROGRESS_MONITOR_CUH

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include "uint128_improved.cuh"

// Progress tracking structure for device
struct ProgressState {
    // Iteration tracking (use regular types for device compatibility)
    unsigned long long total_iterations;
    unsigned long long successful_iterations;
    unsigned int active_threads;
    
    // Time tracking
    double start_time;
    double last_update_time;
    
    // Performance metrics
    unsigned long long gcd_operations;
    unsigned long long modular_operations;
    
    // Status flags
    bool factor_found;
    bool should_stop;
    
    // GPU metrics (updated periodically)
    float gpu_utilization;
    float gpu_memory_used;
    float gpu_temperature;
    float gpu_power_usage;
    
    // Problem info
    uint128_t target_number;
    int total_threads;
    int update_interval_ms;
    
    __host__ __device__ ProgressState() : 
        total_iterations(0),
        successful_iterations(0),
        active_threads(0),
        gcd_operations(0),
        modular_operations(0),
        factor_found(false),
        should_stop(false),
        gpu_utilization(0.0f),
        gpu_memory_used(0.0f),
        gpu_temperature(0.0f),
        gpu_power_usage(0.0f),
        total_threads(0),
        update_interval_ms(1000) {
        start_time = 0.0;
        last_update_time = 0.0;
    }
    
    static double get_current_time() {
        return (double)clock() / CLOCKS_PER_SEC;
    }
};

// Simple GPU metrics placeholder (NVML requires host-only implementation)
class GPUMonitor {
private:
    bool initialized;
    
public:
    GPUMonitor() : initialized(false) {
        // Simplified - real NVML implementation would go here
    }
    
    ~GPUMonitor() {
    }
    
    void update_metrics(ProgressState* state) {
        // Placeholder - real implementation would use NVML
        state->gpu_utilization = 75.0f;  // Mock value
        state->gpu_memory_used = 50.0f;   // Mock value
        state->gpu_temperature = 65.0f;   // Mock value
        state->gpu_power_usage = 120.0f;  // Mock value
    }
};

// Device-side progress update function
__device__ void update_progress_device(
    ProgressState* progress,
    int iterations_done,
    int gcd_ops = 0,
    int mod_ops = 0
) {
    atomicAdd((unsigned long long*)&progress->total_iterations, iterations_done);
    atomicAdd((unsigned long long*)&progress->gcd_operations, gcd_ops);
    atomicAdd((unsigned long long*)&progress->modular_operations, mod_ops);
}

// Host-side progress reporter
class ProgressReporter {
private:
    ProgressState* d_progress;  // Device memory
    ProgressState h_progress;   // Host copy
    GPUMonitor gpu_monitor;
    FILE* log_file;
    bool verbose;
    
public:
    ProgressReporter(uint128_t target, int total_threads, bool verbose = true) 
        : verbose(verbose), log_file(nullptr) {
        
        // Allocate device memory for progress state
        cudaMalloc(&d_progress, sizeof(ProgressState));
        
        // Initialize host state
        h_progress = ProgressState();
        h_progress.target_number = target;
        h_progress.total_threads = total_threads;
        h_progress.start_time = ProgressState::get_current_time();
        h_progress.last_update_time = h_progress.start_time;
        
        // Copy initial state to device
        cudaMemcpy(d_progress, &h_progress, sizeof(ProgressState), cudaMemcpyHostToDevice);
        
        // Open log file
        if (verbose) {
            log_file = fopen("factorization_progress.log", "w");
            if (log_file) {
                fprintf(log_file, "Time,Iterations,Rate,GPU%%,Memory%%,Temp,Power\n");
            }
        }
    }
    
    ~ProgressReporter() {
        if (d_progress) cudaFree(d_progress);
        if (log_file) fclose(log_file);
    }
    
    ProgressState* get_device_pointer() { return d_progress; }
    
    void update_and_report() {
        // Copy progress from device
        cudaMemcpy(&h_progress, d_progress, sizeof(ProgressState), cudaMemcpyDeviceToHost);
        
        // Update GPU metrics
        gpu_monitor.update_metrics(&h_progress);
        
        double current_time = ProgressState::get_current_time();
        double elapsed = current_time - h_progress.start_time;
        
        if (verbose && (current_time - h_progress.last_update_time) >= 
            (h_progress.update_interval_ms / 1000.0)) {
            
            print_progress(elapsed);
            h_progress.last_update_time = current_time;
        }
    }
    
    void print_progress(double elapsed) {
        // Calculate rates
        double iterations_per_sec = h_progress.total_iterations / elapsed;
        double gcd_per_sec = h_progress.gcd_operations / elapsed;
        
        // Estimate time to completion (heuristic)
        double progress_rate = (h_progress.successful_iterations > 0) ? 
            (double)h_progress.successful_iterations / (double)h_progress.total_iterations : 0.0;
        double eta = (progress_rate > 0.001) ? 
            elapsed / progress_rate - elapsed : -1;
        
        // Console output
        printf("\n===== Factorization Progress =====\n");
        printf("Target: %llx:%llx\n", h_progress.target_number.high, h_progress.target_number.low);
        printf("Elapsed: %.2f seconds\n", elapsed);
        printf("Total iterations: %llu (%.2f M/sec)\n", 
               h_progress.total_iterations, 
               iterations_per_sec / 1e6);
        printf("GCD operations: %llu (%.2f M/sec)\n", 
               h_progress.gcd_operations,
               gcd_per_sec / 1e6);
        printf("Active threads: %u / %d\n", 
               h_progress.active_threads, 
               h_progress.total_threads);
        
        if (eta > 0 && eta < 3600) {  // Only show if reasonable
            printf("Estimated time remaining: %.2f seconds\n", eta);
        }
        
        printf("\n--- GPU Metrics ---\n");
        printf("GPU Utilization: %.1f%%\n", h_progress.gpu_utilization);
        printf("Memory Usage: %.1f%%\n", h_progress.gpu_memory_used);
        printf("Temperature: %.1f°C\n", h_progress.gpu_temperature);
        printf("Power Usage: %.1f W\n", h_progress.gpu_power_usage);
        printf("================================\n\n");
        
        // Log to file
        if (log_file) {
            fprintf(log_file, "%.2f,%llu,%.2f,%.1f,%.1f,%.1f,%.1f\n",
                elapsed,
                h_progress.total_iterations,
                iterations_per_sec / 1e6,
                h_progress.gpu_utilization,
                h_progress.gpu_memory_used,
                h_progress.gpu_temperature,
                h_progress.gpu_power_usage);
            fflush(log_file);
        }
    }
    
    bool should_stop() {
        return h_progress.factor_found || h_progress.should_stop;
    }
    
    void signal_factor_found() {
        h_progress.factor_found = true;
        cudaMemcpy(&d_progress->factor_found, &h_progress.factor_found, 
                   sizeof(bool), cudaMemcpyHostToDevice);
    }
};

// Integration with Pollard's Rho kernel
__global__ void pollards_rho_with_progress(
    uint128_t n,
    uint128_t* factors,
    int* factor_count,
    ProgressState* progress,
    int max_iterations = 1000000
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Mark thread as active
    if (threadIdx.x == 0) {
        atomicAdd(&progress->active_threads, blockDim.x);
    }
    __syncthreads();
    
    // Main factorization loop
    int local_iterations = 0;
    int local_gcd_ops = 0;
    int local_mod_ops = 0;
    
    // Simplified Pollard's Rho for testing
    uint128_t x(tid + 2, 0);
    uint128_t y = x;
    uint128_t factor(1, 0);
    
    for (int i = 0; i < max_iterations && !progress->should_stop && !progress->factor_found; i++) {
        // Simulate work
        x = add_128(multiply_64_64(x.low, x.low), uint128_t(1, 0));
        if (x >= n) x = subtract_128(x, n);
        local_mod_ops++;
        
        y = add_128(multiply_64_64(y.low, y.low), uint128_t(1, 0));
        if (y >= n) y = subtract_128(y, n);
        y = add_128(multiply_64_64(y.low, y.low), uint128_t(1, 0));
        if (y >= n) y = subtract_128(y, n);
        local_mod_ops += 2;
        
        uint128_t diff = (x > y) ? subtract_128(x, y) : subtract_128(y, x);
        factor = gcd_128(diff, n);
        local_gcd_ops++;
        
        local_iterations++;
        
        // Update progress periodically
        if (local_iterations % 100 == 0) {
            update_progress_device(progress, 100, local_gcd_ops, local_mod_ops);
            local_iterations = 0;
            local_gcd_ops = 0;
            local_mod_ops = 0;
        }
        
        // Check if factor found
        if (factor.low > 1 && factor < n) {
            int idx = atomicAdd(factor_count, 1);
            if (idx < 32) {  // MAX_FACTORS
                factors[idx] = factor;
                progress->factor_found = true;
            }
            break;
        }
    }
    
    // Update any remaining progress
    if (local_iterations > 0) {
        update_progress_device(progress, local_iterations, local_gcd_ops, local_mod_ops);
    }
    
    // Mark thread as inactive
    if (threadIdx.x == 0) {
        atomicSub(&progress->active_threads, blockDim.x);
    }
}

// Performance analysis kernel
__global__ void benchmark_with_monitoring(ProgressState* progress) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Simulate workload
    uint128_t x(tid, tid);
    uint128_t n(0x123456789ABCDEFULL, 0x1ULL);
    
    clock_t start = clock();
    
    for (int i = 0; i < 10000; i++) {
        // Simulate GCD operation
        uint128_t g = gcd_128(x, n);
        update_progress_device(progress, 1, 1, 0);
        
        // Simulate modular operation
        x = add_128(x, g);
        if (x >= n) x = subtract_128(x, n);
        update_progress_device(progress, 0, 0, 1);
    }
    
    clock_t end = clock();
    
    if (tid == 0) {
        double time_ms = (double)(end - start) / (double)CLOCKS_PER_SEC * 1000.0;
        printf("Benchmark: 10000 operations in %.3f ms\n", time_ms);
    }
}

// Helper function to run factorization with progress monitoring
void factorize_with_progress(uint128_t n, int num_blocks = 32, int threads_per_block = 256) {
    // Allocate memory for factors
    uint128_t* d_factors;
    int* d_factor_count;
    cudaMalloc(&d_factors, 32 * sizeof(uint128_t));  // MAX_FACTORS = 32
    cudaMalloc(&d_factor_count, sizeof(int));
    cudaMemset(d_factor_count, 0, sizeof(int));
    
    // Create progress reporter
    ProgressReporter reporter(n, num_blocks * threads_per_block);
    
    // Launch kernel
    pollards_rho_with_progress<<<num_blocks, threads_per_block>>>(
        n, d_factors, d_factor_count, reporter.get_device_pointer()
    );
    
    // Monitor progress
    while (!reporter.should_stop()) {
        reporter.update_and_report();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // Check if kernel finished
        if (cudaStreamQuery(0) == cudaSuccess) {
            break;
        }
        
        // Sleep briefly to avoid excessive polling
        usleep(100000); // 100ms
    }
    
    // Final report
    reporter.update_and_report();
    
    // Copy results
    int h_factor_count;
    cudaMemcpy(&h_factor_count, d_factor_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_factor_count > 0) {
        uint128_t* h_factors = new uint128_t[h_factor_count];
        cudaMemcpy(h_factors, d_factors, h_factor_count * sizeof(uint128_t), cudaMemcpyDeviceToHost);
        
        printf("\nFactors found:\n");
        for (int i = 0; i < h_factor_count; i++) {
            printf("  %llx:%llx\n", h_factors[i].high, h_factors[i].low);
        }
        
        delete[] h_factors;
    }
    
    // Cleanup
    cudaFree(d_factors);
    cudaFree(d_factor_count);
}

#endif // PROGRESS_MONITOR_CUH