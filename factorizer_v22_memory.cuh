/**
 * Factorizer v2.2.0 - Memory Management Module
 * 
 * Efficient memory pool implementation for GPU and host memory
 * with support for different allocation strategies.
 */

#ifndef FACTORIZER_V22_MEMORY_CUH
#define FACTORIZER_V22_MEMORY_CUH

#include "factorizer_v22_architecture.h"
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <vector>

//=============================================================================
// Memory Pool Implementation
//=============================================================================

class MemoryPool {
private:
    // Block size categories for pooling
    static constexpr size_t BLOCK_SIZES[] = {
        64,          // Small allocations
        256,
        1024,        // 1 KB
        4096,        // 4 KB
        16384,       // 16 KB
        65536,       // 64 KB
        262144,      // 256 KB
        1048576,     // 1 MB
        4194304,     // 4 MB
        16777216,    // 16 MB
        67108864,    // 64 MB
        268435456    // 256 MB
    };
    static constexpr int NUM_BLOCK_SIZES = sizeof(BLOCK_SIZES) / sizeof(BLOCK_SIZES[0]);
    
    // Free list for each block size and memory type
    struct FreeList {
        std::vector<void*> blocks;
        std::mutex mutex;
    };
    
    FreeList free_lists[4][NUM_BLOCK_SIZES]; // [memory_type][block_size_index]
    
    // Active allocations tracking
    struct AllocationInfo {
        size_t size;
        memory_type_t type;
        int block_size_index;
    };
    std::unordered_map<void*, AllocationInfo> allocations;
    std::mutex allocation_mutex;
    
    // Statistics
    size_t total_allocated[4] = {0};
    size_t high_water_mark[4] = {0};
    size_t allocation_count[4] = {0};
    size_t cache_hits[4] = {0};
    
    // CUDA stream for async operations
    cudaStream_t stream;
    
    // Get block size index for allocation
    int get_block_size_index(size_t size) {
        for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
            if (size <= BLOCK_SIZES[i]) {
                return i;
            }
        }
        return -1; // Too large for pooling
    }
    
    // Allocate new block from system
    void* allocate_new_block(size_t size, memory_type_t type) {
        void* ptr = nullptr;
        
        switch (type) {
            case MEM_TYPE_HOST:
                ptr = malloc(size);
                if (!ptr) {
                    throw std::bad_alloc();
                }
                break;
                
            case MEM_TYPE_DEVICE:
                if (cudaMalloc(&ptr, size) != cudaSuccess) {
                    throw std::bad_alloc();
                }
                break;
                
            case MEM_TYPE_UNIFIED:
                if (cudaMallocManaged(&ptr, size) != cudaSuccess) {
                    throw std::bad_alloc();
                }
                break;
                
            case MEM_TYPE_PINNED:
                if (cudaMallocHost(&ptr, size) != cudaSuccess) {
                    throw std::bad_alloc();
                }
                break;
        }
        
        // Update statistics
        total_allocated[type] += size;
        if (total_allocated[type] > high_water_mark[type]) {
            high_water_mark[type] = total_allocated[type];
        }
        allocation_count[type]++;
        
        return ptr;
    }
    
    // Free block back to system
    void free_block(void* ptr, memory_type_t type) {
        switch (type) {
            case MEM_TYPE_HOST:
                free(ptr);
                break;
                
            case MEM_TYPE_DEVICE:
                cudaFree(ptr);
                break;
                
            case MEM_TYPE_UNIFIED:
                cudaFree(ptr);
                break;
                
            case MEM_TYPE_PINNED:
                cudaFreeHost(ptr);
                break;
        }
    }
    
public:
    MemoryPool(cudaStream_t stream = 0) : stream(stream) {}
    
    ~MemoryPool() {
        reset();
    }
    
    /**
     * Allocate memory from pool
     */
    void* allocate(size_t size, memory_type_t type) {
        if (size == 0) return nullptr;
        
        int block_index = get_block_size_index(size);
        
        // Try to get from pool for standard sizes
        if (block_index >= 0) {
            FreeList& list = free_lists[type][block_index];
            
            {
                std::lock_guard<std::mutex> lock(list.mutex);
                if (!list.blocks.empty()) {
                    void* ptr = list.blocks.back();
                    list.blocks.pop_back();
                    
                    // Track allocation
                    {
                        std::lock_guard<std::mutex> lock(allocation_mutex);
                        allocations[ptr] = {BLOCK_SIZES[block_index], type, block_index};
                    }
                    
                    cache_hits[type]++;
                    return ptr;
                }
            }
            
            // Allocate new block of standard size
            size = BLOCK_SIZES[block_index];
        }
        
        // Allocate new block
        void* ptr = allocate_new_block(size, type);
        
        // Track allocation
        {
            std::lock_guard<std::mutex> lock(allocation_mutex);
            allocations[ptr] = {size, type, block_index};
        }
        
        return ptr;
    }
    
    /**
     * Free memory back to pool
     */
    void free(void* ptr) {
        if (!ptr) return;
        
        AllocationInfo info;
        
        // Find allocation info
        {
            std::lock_guard<std::mutex> lock(allocation_mutex);
            auto it = allocations.find(ptr);
            if (it == allocations.end()) {
                // Not our allocation
                return;
            }
            info = it->second;
            allocations.erase(it);
        }
        
        // Update statistics
        total_allocated[info.type] -= info.size;
        
        // Return to pool if standard size
        if (info.block_size_index >= 0) {
            FreeList& list = free_lists[info.type][info.block_size_index];
            
            std::lock_guard<std::mutex> lock(list.mutex);
            list.blocks.push_back(ptr);
        } else {
            // Free directly for non-standard sizes
            free_block(ptr, info.type);
        }
    }
    
    /**
     * Reset pool and free all memory
     */
    void reset() {
        // Free all pooled blocks
        for (int type = 0; type < 4; type++) {
            for (int size = 0; size < NUM_BLOCK_SIZES; size++) {
                FreeList& list = free_lists[type][size];
                std::lock_guard<std::mutex> lock(list.mutex);
                
                for (void* ptr : list.blocks) {
                    free_block(ptr, (memory_type_t)type);
                }
                list.blocks.clear();
            }
        }
        
        // Free any remaining allocations
        {
            std::lock_guard<std::mutex> lock(allocation_mutex);
            for (const auto& pair : allocations) {
                free_block(pair.first, pair.second.type);
            }
            allocations.clear();
        }
        
        // Reset statistics
        for (int i = 0; i < 4; i++) {
            total_allocated[i] = 0;
        }
    }
    
    /**
     * Get memory usage statistics
     */
    size_t get_usage(memory_type_t type) const {
        return total_allocated[type];
    }
    
    size_t get_high_water_mark(memory_type_t type) const {
        return high_water_mark[type];
    }
    
    void print_statistics() const {
        const char* type_names[] = {"Host", "Device", "Unified", "Pinned"};
        
        printf("\n=== Memory Pool Statistics ===\n");
        for (int i = 0; i < 4; i++) {
            if (allocation_count[i] > 0) {
                printf("%s Memory:\n", type_names[i]);
                printf("  Current: %.2f MB\n", total_allocated[i] / (1024.0 * 1024.0));
                printf("  Peak:    %.2f MB\n", high_water_mark[i] / (1024.0 * 1024.0));
                printf("  Allocations: %zu\n", allocation_count[i]);
                printf("  Cache hits:  %zu (%.1f%%)\n", 
                    cache_hits[i], 
                    100.0 * cache_hits[i] / allocation_count[i]);
            }
        }
        printf("==============================\n");
    }
};

//=============================================================================
// Specialized Memory Allocators
//=============================================================================

/**
 * Scratch memory allocator for temporary computations
 * Uses a simple bump allocator that can be reset quickly
 */
class ScratchAllocator {
private:
    uint8_t* base_ptr;
    size_t capacity;
    size_t offset;
    memory_type_t type;
    MemoryPool* pool;
    
public:
    ScratchAllocator(MemoryPool* pool, size_t size, memory_type_t type) 
        : pool(pool), capacity(size), offset(0), type(type) {
        base_ptr = (uint8_t*)pool->allocate(size, type);
    }
    
    ~ScratchAllocator() {
        if (base_ptr) {
            pool->free(base_ptr);
        }
    }
    
    void* allocate(size_t size, size_t alignment = 16) {
        // Align offset
        size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
        
        if (aligned_offset + size > capacity) {
            return nullptr; // Out of space
        }
        
        void* ptr = base_ptr + aligned_offset;
        offset = aligned_offset + size;
        return ptr;
    }
    
    void reset() {
        offset = 0;
    }
    
    size_t used() const {
        return offset;
    }
    
    size_t available() const {
        return capacity - offset;
    }
};

//=============================================================================
// Memory Manager Interface Implementation
//=============================================================================

// C-style interface functions for architecture compatibility
extern "C" {

void* memory_manager_allocate(memory_pool_t* pool, size_t size, memory_type_t type) {
    return static_cast<MemoryPool*>(pool)->allocate(size, type);
}

void memory_manager_free(memory_pool_t* pool, void* ptr) {
    static_cast<MemoryPool*>(pool)->free(ptr);
}

void memory_manager_reset(memory_pool_t* pool) {
    static_cast<MemoryPool*>(pool)->reset();
}

void memory_manager_destroy(memory_pool_t* pool) {
    delete static_cast<MemoryPool*>(pool);
}

size_t memory_manager_get_usage(memory_pool_t* pool, memory_type_t type) {
    return static_cast<MemoryPool*>(pool)->get_usage(type);
}

memory_manager_t* create_memory_manager() {
    static memory_manager_t manager = {
        .allocate = memory_manager_allocate,
        .free = memory_manager_free,
        .reset = memory_manager_reset,
        .destroy = memory_manager_destroy,
        .get_usage = memory_manager_get_usage
    };
    return &manager;
}

} // extern "C"

//=============================================================================
// GPU Memory Utilities
//=============================================================================

/**
 * Query available GPU memory
 */
inline bool query_gpu_memory(size_t* free_bytes, size_t* total_bytes) {
    return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess;
}

/**
 * Estimate memory requirements for factorization
 */
struct MemoryRequirements {
    size_t host_memory;
    size_t device_memory;
    size_t unified_memory;
    size_t pinned_memory;
    
    size_t total() const {
        return host_memory + device_memory + unified_memory + pinned_memory;
    }
    
    bool fits_in_gpu(size_t available_gpu_memory) const {
        return device_memory + unified_memory <= available_gpu_memory;
    }
};

/**
 * Estimate memory requirements for different algorithms
 */
MemoryRequirements estimate_memory_requirements(
    algorithm_id_t algorithm,
    const factor_t* number,
    int thread_count
) {
    MemoryRequirements req = {0};
    int digits = count_digits(number);
    
    switch (algorithm) {
        case ALGO_POLLARD_RHO:
            // Per-thread state: x, y, c values (3 * 16 bytes)
            req.device_memory = thread_count * 48;
            // Shared factor storage
            req.device_memory += 1024;
            // Host-side monitoring
            req.pinned_memory = 4096;
            break;
            
        case ALGO_ECM:
            // Elliptic curve points (multiple per thread)
            req.device_memory = thread_count * 1024;
            // Precomputed values
            req.device_memory += (1 << 20); // 1 MB
            // Stage 2 baby-step giant-step tables
            req.device_memory += digits * (1 << 16); // Scales with input size
            break;
            
        case ALGO_QUADRATIC_SIEVE:
            // Sieving array
            size_t sieve_size = (size_t)pow(2, digits * 0.5) * 1024;
            req.device_memory = sieve_size;
            // Factor base
            req.device_memory += digits * 10000;
            // Matrix for linear algebra
            req.host_memory = (size_t)pow(digits, 2) * 1000;
            break;
            
        default:
            // Conservative estimate
            req.device_memory = 256 << 20; // 256 MB
            req.host_memory = 128 << 20;   // 128 MB
            break;
    }
    
    return req;
}

//=============================================================================
// Memory-Aware Algorithm Configuration
//=============================================================================

/**
 * Adjust algorithm parameters based on available memory
 */
void configure_algorithm_for_memory(
    algorithm_params_t* params,
    const MemoryRequirements& requirements,
    size_t available_gpu_memory
) {
    // If we don't fit in GPU memory, reduce parallelism
    if (requirements.device_memory > available_gpu_memory) {
        float reduction_factor = (float)available_gpu_memory / requirements.device_memory;
        
        // Reduce thread count
        params->thread_count = (int)(params->thread_count * reduction_factor * 0.9f);
        params->thread_count = (params->thread_count / 32) * 32; // Round to warp size
        params->thread_count = max(32, params->thread_count); // Minimum 1 warp
        
        // Adjust block count accordingly
        params->block_count = (params->thread_count + 255) / 256;
    }
    
    // For memory-intensive algorithms, may need to adjust algorithm-specific params
    if (params->algorithm_specific) {
        // This would be handled by algorithm-specific configuration
    }
}

#endif // FACTORIZER_V22_MEMORY_CUH