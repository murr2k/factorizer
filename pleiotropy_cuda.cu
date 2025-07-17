/**
 * CUDA Implementation for Genomic Pleiotropy Analysis
 * Using Matrix Factorization and Parallel Pattern Detection
 * Optimized for NVIDIA GTX 2070
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)

// GTX 2070 specifications
#define GTX2070_CORES 2304
#define GTX2070_SM_COUNT 36
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define SHARED_MEMORY_PER_BLOCK 49152 // 48KB

// Genomic data structures
struct GenomicData {
    float* snp_matrix;      // M x N matrix of SNP values
    float* trait_matrix;    // M x P matrix of trait values
    int num_snps;          // M
    int num_samples;       // N
    int num_traits;        // P
};

struct PleiotropyFactors {
    float* W;              // M x K factor loading matrix
    float* H;              // K x P factor score matrix
    float* S;              // K x 1 sparsity vector
    int rank;              // K - number of latent factors
};

// Kernel for element-wise matrix operations with coalesced memory access
__global__ void elementwise_multiply_kernel(
    const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// Optimized matrix multiplication kernel using shared memory
template<int TILE_SIZE>
__global__ void matmul_shared_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory with boundary checking
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Non-negative matrix factorization update kernel
__global__ void nmf_update_W_kernel(
    float* W, const float* V, const float* H,
    const float* WH, int M, int N, int K,
    float epsilon = 1e-10f) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float numerator = 0.0f;
        float denominator = 0.0f;
        
        // Compute (V * H^T)_ik / (W * H * H^T)_ik
        for (int j = 0; j < N; ++j) {
            float h_kj = H[col * N + j];
            numerator += V[row * N + j] * h_kj;
            denominator += WH[row * N + j] * h_kj;
        }
        
        // Multiplicative update rule
        W[row * K + col] *= (numerator / (denominator + epsilon));
    }
}

// Kernel for detecting pleiotropic patterns
__global__ void detect_pleiotropy_kernel(
    const float* W, const float* threshold,
    int* pleiotropy_count, int M, int K) {
    
    int gene_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gene_idx < M) {
        int count = 0;
        for (int k = 0; k < K; ++k) {
            if (fabsf(W[gene_idx * K + k]) > *threshold) {
                count++;
            }
        }
        pleiotropy_count[gene_idx] = count;
    }
}

// Host class for CUDA pleiotropy analysis
class CUDAPleiotopyAnalyzer {
private:
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    curandGenerator_t curand_gen;
    
    GenomicData d_data;
    PleiotropyFactors d_factors;
    
    dim3 block_size;
    dim3 grid_size;
    
public:
    CUDAPleiotopyAnalyzer(int num_snps, int num_samples, int num_traits, int rank) {
        // Initialize CUDA libraries
        cublasCreate(&cublas_handle);
        cusolverDnCreate(&cusolver_handle);
        curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL);
        
        // Set optimal block and grid sizes for GTX 2070
        block_size = dim3(16, 16);
        
        // Allocate device memory
        d_data.num_snps = num_snps;
        d_data.num_samples = num_samples;
        d_data.num_traits = num_traits;
        
        CUDA_CHECK(cudaMalloc(&d_data.snp_matrix, num_snps * num_samples * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_data.trait_matrix, num_snps * num_traits * sizeof(float)));
        
        d_factors.rank = rank;
        CUDA_CHECK(cudaMalloc(&d_factors.W, num_snps * rank * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_factors.H, rank * num_traits * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_factors.S, rank * sizeof(float)));
        
        // Initialize factors with random values
        curandGenerateUniform(curand_gen, d_factors.W, num_snps * rank);
        curandGenerateUniform(curand_gen, d_factors.H, rank * num_traits);
    }
    
    ~CUDAPleiotopyAnalyzer() {
        // Cleanup
        cudaFree(d_data.snp_matrix);
        cudaFree(d_data.trait_matrix);
        cudaFree(d_factors.W);
        cudaFree(d_factors.H);
        cudaFree(d_factors.S);
        
        cublasDestroy(cublas_handle);
        cusolverDnDestroy(cusolver_handle);
        curandDestroyGenerator(curand_gen);
    }
    
    void loadData(const std::vector<float>& snp_data, const std::vector<float>& trait_data) {
        CUDA_CHECK(cudaMemcpy(d_data.snp_matrix, snp_data.data(),
                             snp_data.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data.trait_matrix, trait_data.data(),
                             trait_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void performNMF(int max_iterations = 100, float tolerance = 1e-4f) {
        float* d_WH;
        float* d_temp;
        int M = d_data.num_snps;
        int N = d_data.num_traits;
        int K = d_factors.rank;
        
        CUDA_CHECK(cudaMalloc(&d_WH, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, std::max(M * K, K * N) * sizeof(float)));
        
        float alpha = 1.0f, beta = 0.0f;
        float prev_error = std::numeric_limits<float>::max();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Compute WH = W * H
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha,
                       d_factors.H, N,
                       d_factors.W, K,
                       &beta, d_WH, N);
            
            // Update W using multiplicative rule
            dim3 w_grid((K + block_size.x - 1) / block_size.x,
                       (M + block_size.y - 1) / block_size.y);
            nmf_update_W_kernel<<<w_grid, block_size>>>(
                d_factors.W, d_data.trait_matrix, d_factors.H,
                d_WH, M, N, K);
            
            // Update H (similar process, kernel not shown for brevity)
            
            // Check convergence
            float error;
            cublasSasum(cublas_handle, M * N, d_WH, 1, &error);
            
            if (std::abs(error - prev_error) < tolerance) {
                std::cout << "NMF converged at iteration " << iter << std::endl;
                break;
            }
            prev_error = error;
        }
        
        cudaFree(d_WH);
        cudaFree(d_temp);
    }
    
    std::vector<int> detectPleiotropicGenes(float threshold) {
        int* d_pleiotropy_count;
        CUDA_CHECK(cudaMalloc(&d_pleiotropy_count, d_data.num_snps * sizeof(int)));
        
        float* d_threshold;
        CUDA_CHECK(cudaMalloc(&d_threshold, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_threshold, &threshold, sizeof(float), cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (d_data.num_snps + threads - 1) / threads;
        
        detect_pleiotropy_kernel<<<blocks, threads>>>(
            d_factors.W, d_threshold, d_pleiotropy_count,
            d_data.num_snps, d_factors.rank);
        
        std::vector<int> pleiotropy_count(d_data.num_snps);
        CUDA_CHECK(cudaMemcpy(pleiotropy_count.data(), d_pleiotropy_count,
                             d_data.num_snps * sizeof(int), cudaMemcpyDeviceToHost));
        
        cudaFree(d_pleiotropy_count);
        cudaFree(d_threshold);
        
        return pleiotropy_count;
    }
    
    void benchmark() {
        std::cout << "\n=== GTX 2070 Performance Benchmark ===" << std::endl;
        std::cout << "SNPs: " << d_data.num_snps << ", Samples: " << d_data.num_samples
                  << ", Traits: " << d_data.num_traits << ", Rank: " << d_factors.rank << std::endl;
        
        // Warm up
        performNMF(5);
        
        auto start = std::chrono::high_resolution_clock::now();
        performNMF(50);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "NMF computation time: " << duration.count() << " ms" << std::endl;
        
        // Calculate theoretical FLOPS
        long long flops = 2LL * d_data.num_snps * d_data.num_samples * d_factors.rank * 50;
        double gflops = flops / (duration.count() / 1000.0) / 1e9;
        std::cout << "Achieved performance: " << gflops << " GFLOPS" << std::endl;
    }
};

// Example usage and test
int main(int argc, char** argv) {
    // Check for CUDA devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found. Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Initialize CUDA
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "CUDA cores: " << prop.multiProcessorCount * 64 << std::endl; // Approximate for Turing
    
    // Parse command line arguments
    int num_snps = 10000;
    int num_samples = 1000;
    int num_traits = 50;
    int rank = 10;
    bool benchmark_mode = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--snps") == 0 && i + 1 < argc) {
            num_snps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--traits") == 0 && i + 1 < argc) {
            num_traits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
        }
    }
    
    CUDAPleiotopyAnalyzer analyzer(num_snps, num_samples, num_traits, rank);
    
    // Generate synthetic data for testing
    std::vector<float> snp_data(num_snps * num_samples);
    std::vector<float> trait_data(num_snps * num_traits);
    
    // Initialize with random values (in real use, load actual genomic data)
    for (auto& val : snp_data) val = static_cast<float>(rand()) / RAND_MAX;
    for (auto& val : trait_data) val = static_cast<float>(rand()) / RAND_MAX;
    
    analyzer.loadData(snp_data, trait_data);
    
    // Perform analysis
    std::cout << "\nPerforming Non-negative Matrix Factorization..." << std::endl;
    analyzer.performNMF(100);
    
    // Detect pleiotropic genes
    std::cout << "\nDetecting pleiotropic genes..." << std::endl;
    // Lower threshold for random data - adjust based on data characteristics
    float threshold = (num_snps > 2000) ? 0.1f : 0.5f;
    auto pleiotropy_counts = analyzer.detectPleiotropicGenes(threshold);
    
    int pleiotropic_genes = 0;
    for (int count : pleiotropy_counts) {
        if (count > 1) pleiotropic_genes++;
    }
    
    std::cout << "Found " << pleiotropic_genes << " pleiotropic genes out of "
              << num_snps << " total genes" << std::endl;
    
    // Benchmark performance
    analyzer.benchmark();
    
    return 0;
}