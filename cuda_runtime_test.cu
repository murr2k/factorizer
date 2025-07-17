#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel
__global__ void testKernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    printf("CUDA Runtime Test\n");
    
    // Initialize runtime
    cudaError_t err = cudaFree(0);
    printf("cudaFree(0): %s\n", cudaGetErrorString(err));
    
    // Get device count
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount: %s, count=%d\n", cudaGetErrorString(err), deviceCount);
    
    if (deviceCount > 0) {
        // Launch a simple kernel
        testKernel<<<1, 4>>>();
        err = cudaDeviceSynchronize();
        printf("Kernel launch: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}