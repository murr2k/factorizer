#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Testing CUDA device detection..." << std::endl;
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    std::cout << "cudaGetDeviceCount returned: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Device count: " << deviceCount << std::endl;
    
    if (err == cudaSuccess && deviceCount > 0) {
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        }
    }
    
    // Try to set device
    if (deviceCount > 0) {
        err = cudaSetDevice(0);
        std::cout << "\ncudaSetDevice(0) returned: " << cudaGetErrorString(err) << std::endl;
    }
    
    return 0;
}