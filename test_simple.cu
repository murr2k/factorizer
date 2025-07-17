/**
 * Ultra-simple CUDA test
 */

#include <cuda_runtime.h>
#include <cstdio>

__global__ void simple_kernel(int* result) {
    *result = 42;
}

int main() {
    printf("Simple CUDA Test\n");
    printf("================\n");
    
    // Check CUDA
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err \!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("CUDA devices found: %d\n", device_count);
    
    // Simple kernel test
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    
    simple_kernel<<<1, 1>>>(d_result);
    
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Kernel result: %d (expected 42)\n", h_result);
    
    cudaFree(d_result);
    
    printf("Test completed successfully\!\n");
    
    return 0;
}
EOF < /dev/null
