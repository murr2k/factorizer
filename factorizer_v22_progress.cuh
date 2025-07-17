/**
 * Factorizer v2.2.0 - Progress Tracking Module
 * 
 * Real-time progress monitoring and ETA calculation for
 * long-running factorization operations.
 */

#ifndef FACTORIZER_V22_PROGRESS_CUH
#define FACTORIZER_V22_PROGRESS_CUH

#include "factorizer_v22_architecture.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>

//=============================================================================
// Progress Tracker Implementation
//=============================================================================

class ProgressTracker {
private:
    progress_info_t info;
    volatile int* device_counter;
    int* host_counter_mirror;
    cudaEvent_t start_event;
    cudaEvent_t current_event;
    cudaStream_t stream;
    
    // Progress callback
    void (*callback)(const progress_info_t* info, void* user_data);
    void* callback_data;
    
    // Update control
    std::chrono::steady_clock::time_point last_update;
    int update_interval_ms;
    std::atomic<bool> stop_flag;
    std::thread* monitor_thread;
    
    // ETA calculation
    struct ETACalculator {
        static constexpr int HISTORY_SIZE = 10;
        double iteration_rates[HISTORY_SIZE];
        int history_index;
        int history_count;
        
        ETACalculator() : history_index(0), history_count(0) {}
        
        void add_rate(double rate) {
            iteration_rates[history_index] = rate;
            history_index = (history_index + 1) % HISTORY_SIZE;
            if (history_count < HISTORY_SIZE) history_count++;
        }
        
        double get_smoothed_rate() {
            if (history_count == 0) return 0.0;
            
            double sum = 0.0;
            double weight_sum = 0.0;
            
            // Weighted average with more weight on recent values
            for (int i = 0; i < history_count; i++) {
                int idx = (history_index - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
                double weight = exp(-i * 0.2); // Exponential decay
                sum += iteration_rates[idx] * weight;
                weight_sum += weight;
            }
            
            return sum / weight_sum;
        }
    } eta_calculator;
    
    // Algorithm-specific progress estimation
    double estimate_total_iterations(algorithm_id_t algo, const factor_t* number) {
        int digits = count_digits(number);
        
        switch (algo) {
            case ALGO_POLLARD_RHO:
                // Expected iterations: O(sqrt(p)) where p is smallest factor
                // Estimate p ≈ sqrt(n) for semiprime
                return pow(10, digits / 4.0) * 1.25; // With safety factor
                
            case ALGO_ECM:
                // Depends on curve count and bounds
                return 1000.0 * pow(digits, 2); // Rough estimate
                
            case ALGO_QUADRATIC_SIEVE:
                // Sieving operations
                return pow(2, digits * 0.5) * 1000;
                
            default:
                return 1e9; // Conservative estimate
        }
    }
    
    // Monitor thread function
    void monitor_progress() {
        while (!stop_flag.load()) {
            update();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(update_interval_ms)
            );
        }
    }
    
public:
    ProgressTracker(cudaStream_t stream = 0) 
        : stream(stream), 
          device_counter(nullptr),
          host_counter_mirror(nullptr),
          callback(nullptr),
          callback_data(nullptr),
          update_interval_ms(100),
          stop_flag(false),
          monitor_thread(nullptr) {
        
        // Initialize events
        cudaEventCreate(&start_event);
        cudaEventCreate(&current_event);
        
        // Initialize info
        memset(&info, 0, sizeof(info));
        strcpy(info.status_message, "Initializing...");
    }
    
    ~ProgressTracker() {
        stop();
        
        if (device_counter) cudaFree((void*)device_counter);
        if (host_counter_mirror) cudaFreeHost(host_counter_mirror);
        
        cudaEventDestroy(start_event);
        cudaEventDestroy(current_event);
    }
    
    /**
     * Initialize progress tracking for an algorithm
     */
    void initialize(algorithm_id_t algo, const factor_t* number) {
        // Allocate device counter
        if (!device_counter) {
            cudaMalloc((void**)&device_counter, sizeof(int));
            cudaMemset((void*)device_counter, 0, sizeof(int));
            
            // Allocate pinned host mirror for fast access
            cudaMallocHost(&host_counter_mirror, sizeof(int));
            *host_counter_mirror = 0;
        }
        
        // Reset info
        info.iterations_completed = 0;
        info.iterations_total = (uint64_t)estimate_total_iterations(algo, number);
        info.percentage = 0.0f;
        info.elapsed_seconds = 0.0;
        info.estimated_remaining = 0.0;
        info.current_algorithm = algo;
        info.factors_found = 0;
        
        // Record start time
        cudaEventRecord(start_event, stream);
        last_update = std::chrono::steady_clock::now();
        
        // Update status
        const char* algo_name = algorithm_database[algo].name;
        snprintf(info.status_message, sizeof(info.status_message),
                "Running %s algorithm...", algo_name);
    }
    
    /**
     * Start monitoring thread
     */
    void start_monitoring() {
        if (!monitor_thread) {
            stop_flag = false;
            monitor_thread = new std::thread(&ProgressTracker::monitor_progress, this);
        }
    }
    
    /**
     * Stop monitoring
     */
    void stop() {
        if (monitor_thread) {
            stop_flag = true;
            monitor_thread->join();
            delete monitor_thread;
            monitor_thread = nullptr;
        }
    }
    
    /**
     * Update progress information
     */
    void update() {
        // Copy device counter to host
        if (device_counter && host_counter_mirror) {
            cudaMemcpyAsync(host_counter_mirror, (void*)device_counter, 
                          sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            info.iterations_completed = *host_counter_mirror;
        }
        
        // Calculate elapsed time
        cudaEventRecord(current_event, stream);
        cudaEventSynchronize(current_event);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_event, current_event);
        info.elapsed_seconds = elapsed_ms / 1000.0;
        
        // Calculate percentage
        if (info.iterations_total > 0) {
            info.percentage = 100.0f * info.iterations_completed / info.iterations_total;
            info.percentage = fminf(info.percentage, 99.9f); // Never show 100% until done
        }
        
        // Calculate iteration rate and ETA
        auto now = std::chrono::steady_clock::now();
        double time_since_last = std::chrono::duration<double>(now - last_update).count();
        
        if (time_since_last > 0.1) { // Update rate calculation
            double current_rate = info.iterations_completed / info.elapsed_seconds;
            eta_calculator.add_rate(current_rate);
            
            double smoothed_rate = eta_calculator.get_smoothed_rate();
            if (smoothed_rate > 0) {
                uint64_t remaining = info.iterations_total - info.iterations_completed;
                info.estimated_remaining = remaining / smoothed_rate;
            }
            
            last_update = now;
        }
        
        // Call user callback
        if (callback) {
            callback(&info, callback_data);
        }
    }
    
    /**
     * Set progress callback
     */
    void set_callback(void (*cb)(const progress_info_t*, void*), void* data) {
        callback = cb;
        callback_data = data;
    }
    
    /**
     * Set update interval
     */
    void set_update_interval(int ms) {
        update_interval_ms = ms;
    }
    
    /**
     * Get device counter pointer for kernels
     */
    volatile int* get_device_counter() {
        return device_counter;
    }
    
    /**
     * Mark completion
     */
    void mark_complete(int factors_found) {
        stop();
        update(); // Final update
        
        info.percentage = 100.0f;
        info.factors_found = factors_found;
        info.estimated_remaining = 0.0;
        
        if (factors_found > 0) {
            snprintf(info.status_message, sizeof(info.status_message),
                    "Factorization complete! Found %d factors.", factors_found);
        } else {
            strcpy(info.status_message, "Factorization complete. Number is likely prime.");
        }
        
        if (callback) {
            callback(&info, callback_data);
        }
    }
    
    /**
     * Get current progress info
     */
    const progress_info_t* get_info() const {
        return &info;
    }
};

//=============================================================================
// GPU Progress Update Utilities
//=============================================================================

/**
 * Device function to update progress counter
 */
__device__ inline void update_progress_counter(volatile int* counter, int increment = 1) {
    if (counter) {
        atomicAdd((int*)counter, increment);
    }
}

/**
 * Device function for batch progress updates
 */
__device__ inline void update_progress_batch(volatile int* counter, int batch_size, int thread_id) {
    // Only one thread per warp updates to reduce contention
    if ((thread_id & 31) == 0) {
        atomicAdd((int*)counter, batch_size);
    }
}

//=============================================================================
// Progress Display Utilities
//=============================================================================

/**
 * Format time duration for display
 */
inline void format_duration(double seconds, char* buffer, size_t size) {
    if (seconds < 60) {
        snprintf(buffer, size, "%.1f seconds", seconds);
    } else if (seconds < 3600) {
        int minutes = (int)(seconds / 60);
        int secs = (int)seconds % 60;
        snprintf(buffer, size, "%d:%02d", minutes, secs);
    } else {
        int hours = (int)(seconds / 3600);
        int minutes = (int)(seconds / 60) % 60;
        int secs = (int)seconds % 60;
        snprintf(buffer, size, "%d:%02d:%02d", hours, minutes, secs);
    }
}

/**
 * Default progress callback - prints to console
 */
void default_progress_callback(const progress_info_t* info, void* user_data) {
    static int last_percentage = -1;
    int current_percentage = (int)info->percentage;
    
    // Only update when percentage changes
    if (current_percentage != last_percentage) {
        last_percentage = current_percentage;
        
        // Clear line and print progress
        printf("\r\033[K"); // Clear line
        
        // Progress bar
        printf("[");
        int bar_width = 50;
        int filled = (int)(bar_width * info->percentage / 100.0f);
        for (int i = 0; i < bar_width; i++) {
            if (i < filled) printf("=");
            else if (i == filled) printf(">");
            else printf(" ");
        }
        printf("] ");
        
        // Percentage and stats
        printf("%3.0f%% ", info->percentage);
        
        // Elapsed time
        char elapsed_str[32];
        format_duration(info->elapsed_seconds, elapsed_str, sizeof(elapsed_str));
        printf("Elapsed: %s ", elapsed_str);
        
        // ETA
        if (info->estimated_remaining > 0 && info->percentage > 1.0f) {
            char eta_str[32];
            format_duration(info->estimated_remaining, eta_str, sizeof(eta_str));
            printf("ETA: %s ", eta_str);
        }
        
        // Iteration rate
        if (info->elapsed_seconds > 0) {
            double rate = info->iterations_completed / info->elapsed_seconds;
            if (rate > 1e9) {
                printf("(%.1f G/s) ", rate / 1e9);
            } else if (rate > 1e6) {
                printf("(%.1f M/s) ", rate / 1e6);
            } else if (rate > 1e3) {
                printf("(%.1f K/s) ", rate / 1e3);
            }
        }
        
        // Status message
        printf("%s", info->status_message);
        
        fflush(stdout);
    }
}

/**
 * Detailed progress callback - includes more information
 */
void detailed_progress_callback(const progress_info_t* info, void* user_data) {
    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    
    // Update every 500ms
    if (std::chrono::duration<double>(now - last_update).count() < 0.5) {
        return;
    }
    last_update = now;
    
    // Clear screen and show detailed info
    printf("\033[2J\033[H"); // Clear screen and move to top
    
    printf("=== Factorizer v2.2.0 Progress ===\n\n");
    
    printf("Algorithm: %s\n", algorithm_database[info->current_algorithm].name);
    printf("Status: %s\n", info->status_message);
    printf("\n");
    
    // Progress bar
    printf("Progress: [");
    int bar_width = 60;
    int filled = (int)(bar_width * info->percentage / 100.0f);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("█");
        else printf("░");
    }
    printf("] %.1f%%\n\n", info->percentage);
    
    // Statistics
    printf("Iterations: %llu / %llu\n", 
           (unsigned long long)info->iterations_completed,
           (unsigned long long)info->iterations_total);
    
    char elapsed_str[32], eta_str[32];
    format_duration(info->elapsed_seconds, elapsed_str, sizeof(elapsed_str));
    printf("Elapsed Time: %s\n", elapsed_str);
    
    if (info->estimated_remaining > 0 && info->percentage > 1.0f) {
        format_duration(info->estimated_remaining, eta_str, sizeof(eta_str));
        printf("Estimated Remaining: %s\n", eta_str);
        
        // Total time estimate
        double total_time = info->elapsed_seconds + info->estimated_remaining;
        format_duration(total_time, eta_str, sizeof(eta_str));
        printf("Estimated Total: %s\n", eta_str);
    }
    
    // Performance metrics
    if (info->elapsed_seconds > 0) {
        double rate = info->iterations_completed / info->elapsed_seconds;
        printf("\nPerformance: ");
        if (rate > 1e9) {
            printf("%.2f billion iterations/sec\n", rate / 1e9);
        } else if (rate > 1e6) {
            printf("%.2f million iterations/sec\n", rate / 1e6);
        } else {
            printf("%.0f iterations/sec\n", rate);
        }
    }
    
    if (info->factors_found > 0) {
        printf("\nFactors found: %d\n", info->factors_found);
    }
    
    printf("\n[Press Ctrl+C to stop]\n");
    fflush(stdout);
}

//=============================================================================
// C Interface Implementation
//=============================================================================

extern "C" {

void factorizer_print_progress(const progress_info_t* info) {
    default_progress_callback(info, nullptr);
    printf("\n");
}

progress_tracker_t* progress_tracker_create(cudaStream_t stream) {
    return reinterpret_cast<progress_tracker_t*>(new ProgressTracker(stream));
}

void progress_tracker_destroy(progress_tracker_t* tracker) {
    delete reinterpret_cast<ProgressTracker*>(tracker);
}

void progress_tracker_initialize(progress_tracker_t* tracker, 
                               algorithm_id_t algo, 
                               const factor_t* number) {
    reinterpret_cast<ProgressTracker*>(tracker)->initialize(algo, number);
}

void progress_tracker_start(progress_tracker_t* tracker) {
    reinterpret_cast<ProgressTracker*>(tracker)->start_monitoring();
}

void progress_tracker_stop(progress_tracker_t* tracker) {
    reinterpret_cast<ProgressTracker*>(tracker)->stop();
}

void progress_tracker_update(progress_tracker_t* tracker) {
    reinterpret_cast<ProgressTracker*>(tracker)->update();
}

volatile int* progress_tracker_get_device_counter(progress_tracker_t* tracker) {
    return reinterpret_cast<ProgressTracker*>(tracker)->get_device_counter();
}

void progress_tracker_mark_complete(progress_tracker_t* tracker, int factors_found) {
    reinterpret_cast<ProgressTracker*>(tracker)->mark_complete(factors_found);
}

} // extern "C"

#endif // FACTORIZER_V22_PROGRESS_CUH