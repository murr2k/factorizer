#ifndef WSL_CUDA_INIT_H
#define WSL_CUDA_INIT_H

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// WSL2-specific CUDA initialization helper
inline bool initializeCUDAForWSL() {
    // Set WSL-specific library path
    const char* wsl_lib_path = "/usr/lib/wsl/lib";
    char* current_ld_path = getenv("LD_LIBRARY_PATH");
    
    if (current_ld_path) {
        std::string new_path = std::string(wsl_lib_path) + ":" + current_ld_path;
        setenv("LD_LIBRARY_PATH", new_path.c_str(), 1);
    } else {
        setenv("LD_LIBRARY_PATH", wsl_lib_path, 1);
    }
    
    // Force CUDA runtime initialization
    cudaFree(0);
    
    // Set device 0 explicitly
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return false;
    }
    
    return true;
}

#endif // WSL_CUDA_INIT_H