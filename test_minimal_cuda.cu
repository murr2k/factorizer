#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Minimal CUDA test" << std::endl;
    
    // Force runtime init
    cudaFree(0);
    
    // Check devices
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    
    std::cout << "Device count: " << count << std::endl;
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << std::endl;
    }
    
    return 0;
}