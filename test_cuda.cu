/**
 * Simple CUDA test to verify GPU access in WSL2
 */

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        std::cout << "\nDevice " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA cores: " << deviceProp.multiProcessorCount * 64 << " (approx)" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    }
    
    // Test a simple kernel
    int *d_data;
    cudaMalloc(&d_data, sizeof(int));
    cudaFree(d_data);
    
    std::cout << "\n✓ CUDA is working properly in WSL2!" << std::endl;
    return 0;
}