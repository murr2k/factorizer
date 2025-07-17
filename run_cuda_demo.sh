#!/bin/bash

# CUDA Demonstration Script for WSL2
# Ensures proper library paths for CUDA runtime

echo "==================================="
echo "CUDA Genomic Pleiotropy Analysis"
echo "Running on NVIDIA GTX 2070"
echo "==================================="

# Set CUDA paths for WSL2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Verify GPU is accessible
echo -e "\n[GPU Status]"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Run the test program first
echo -e "\n[CUDA Test]"
if [ -f "./test_cuda" ]; then
    ./test_cuda
else
    echo "Compiling CUDA test..."
    nvcc -o test_cuda test_cuda.cu
    ./test_cuda
fi

# Now run the actual pleiotropy analyzer
echo -e "\n[Genomic Pleiotropy Analysis Demo]"

# Small scale test
echo -e "\n--- Small Scale Test ---"
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./pleiotropy_analyzer --snps 1000 --samples 100 --traits 10 --rank 5

# If that works, try medium scale
if [ $? -eq 0 ]; then
    echo -e "\n--- Medium Scale Test ---"
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./pleiotropy_analyzer --snps 5000 --samples 500 --traits 50 --rank 10
fi

# Run factorizer demo
echo -e "\n[Factorization Demo]"
if [ -f "./factorizer" ]; then
    echo "Testing genomic sequence factorization..."
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./factorizer --sequence "ATCGATCGATCGATCGATCG"
fi

echo -e "\n✅ CUDA demonstration completed!"