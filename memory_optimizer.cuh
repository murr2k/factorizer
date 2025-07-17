/**
 * Memory optimization utilities for GTX 2070
 * Maximizes bandwidth utilization and minimizes cache misses
 */

#ifndef MEMORY_OPTIMIZER_CUH
#define MEMORY_OPTIMIZER_CUH

#include <cuda_runtime.h>
#include <cuda.h>

// GTX 2070 Memory Specifications
#define GTX2070_GLOBAL_MEMORY_BW 448  // GB/s
#define GTX2070_L2_CACHE_SIZE 4096    // KB
#define GTX2070_L1_CACHE_SIZE 64      // KB per SM
#define GTX2070_SHARED_MEM_SIZE 64     // KB per SM
#define CACHE_LINE_SIZE 128            // bytes

// Aligned memory allocation for coalesced access
template<typename T>
__host__ T* allocate_aligned(size_t num_elements) {
    T* ptr;
    size_t size = num_elements * sizeof(T);
    // Align to 256 bytes for optimal memory transactions
    CUDA_CHECK(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, 0));
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, 0));
    return ptr;
}

// Texture memory wrapper for read-only data
template<typename T>
class TextureArray {
private:
    cudaArray* d_array;
    cudaTextureObject_t tex_obj;
    size_t size;
    
public:
    TextureArray(const T* h_data, size_t num_elements) : size(num_elements) {
        // Create channel descriptor
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
        
        // Allocate array
        CUDA_CHECK(cudaMallocArray(&d_array, &channel_desc, num_elements));
        
        // Copy data to array
        CUDA_CHECK(cudaMemcpy2DToArray(d_array, 0, 0, h_data, 
                                       num_elements * sizeof(T), 
                                       num_elements * sizeof(T), 1,
                                       cudaMemcpyHostToDevice));
        
        // Create texture object
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = d_array;
        
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModePoint;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = false;
        
        CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL));
    }
    
    ~TextureArray() {
        cudaDestroyTextureObject(tex_obj);
        cudaFreeArray(d_array);
    }
    
    __device__ T fetch(int idx) const {
        return tex1Dfetch<T>(tex_obj, idx);
    }
    
    cudaTextureObject_t getTexture() const { return tex_obj; }
};

// Shared memory bank conflict avoidance
template<typename T, int BANK_SIZE = 32>
class BankConflictFreeArray {
private:
    static constexpr int PADDING = 1;
    
public:
    __device__ static int getPaddedIndex(int idx) {
        return idx + (idx / BANK_SIZE) * PADDING;
    }
    
    __device__ static void store(T* shared_mem, int idx, T value) {
        shared_mem[getPaddedIndex(idx)] = value;
    }
    
    __device__ static T load(const T* shared_mem, int idx) {
        return shared_mem[getPaddedIndex(idx)];
    }
};

// Warp-level primitives for efficient reduction
template<typename T>
__device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Memory access pattern analyzer
class MemoryAccessProfiler {
private:
    cudaEvent_t start, stop;
    float elapsed_time;
    
public:
    MemoryAccessProfiler() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~MemoryAccessProfiler() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startProfile() {
        cudaEventRecord(start);
    }
    
    float endProfile() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        return elapsed_time;
    }
    
    float getBandwidthUtilization(size_t bytes_transferred) {
        float bandwidth_gb = (bytes_transferred / 1e9) / (elapsed_time / 1000.0);
        float utilization = (bandwidth_gb / GTX2070_GLOBAL_MEMORY_BW) * 100.0;
        return utilization;
    }
};

// Optimized matrix transpose for coalesced access
template<int TILE_DIM = 32, int BLOCK_ROWS = 8>
__global__ void transpose_coalesced(float* out, const float* in, 
                                   int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Coalesced read from global memory
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // Transpose indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Coalesced write to global memory
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Prefetching strategy for genomic data
template<typename T>
__device__ void prefetch_genomic_data(const T* data, int current_idx, 
                                     int prefetch_distance = 8) {
    // Prefetch future data into L2 cache
    int prefetch_idx = current_idx + prefetch_distance;
    
    // Use inline PTX for cache control
    asm volatile("prefetch.global.L2 [%0];" : : "l"(&data[prefetch_idx]) : "memory");
}

// Optimized memory copy with streaming
template<typename T>
void async_memcpy_optimized(T* dst, const T* src, size_t count, 
                           cudaStream_t stream = 0) {
    // Use pinned memory for optimal transfer
    T* h_pinned;
    CUDA_CHECK(cudaHostAlloc(&h_pinned, count * sizeof(T), cudaHostAllocDefault));
    
    // Copy to pinned memory
    memcpy(h_pinned, src, count * sizeof(T));
    
    // Async copy to device
    CUDA_CHECK(cudaMemcpyAsync(dst, h_pinned, count * sizeof(T), 
                               cudaMemcpyHostToDevice, stream));
    
    // Register callback to free pinned memory after transfer
    auto cleanup = [](cudaStream_t stream, cudaError_t status, void* data) {
        cudaFreeHost(data);
    };
    
    CUDA_CHECK(cudaStreamAddCallback(stream, cleanup, h_pinned, 0));
}

#endif // MEMORY_OPTIMIZER_CUH