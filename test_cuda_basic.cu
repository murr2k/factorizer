#include <cuda_runtime.h>
#include <cstdio>

__global__ void simple_test(int* output) {
    *output = 42;
}

int main() {
    printf("Testing CUDA...\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);
    
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    int* d_output;
    cudaMalloc(&d_output, sizeof(int));
    
    simple_test<<<1, 1>>>(d_output);
    cudaDeviceSynchronize();
    
    int result;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Result: %d\n", result);
    
    cudaFree(d_output);
    
    return 0;
}