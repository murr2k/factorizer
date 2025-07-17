#!/bin/bash

# Simple Test Build Script for CUDA Factorizer v2.2.0

echo "============================================"
echo "  Building Simple Integration Test"
echo "============================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found"
    exit 1
fi

# Set CUDA architecture
CUDA_ARCH="-arch=sm_75"

# Compiler flags
NVCC_FLAGS="-std=c++14 -O2"
INCLUDE_FLAGS="-I."

# Build target
TARGET="test_integrated_simple"

echo "Building simple test program..."
echo "Target: $TARGET"
echo

# Compile
nvcc $CUDA_ARCH $NVCC_FLAGS $INCLUDE_FLAGS \
    -o $TARGET \
    test_integrated_simple.cu

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Executable: ./$TARGET"
    echo
    echo "Usage: ./$TARGET"
    echo
else
    echo "✗ Build failed!"
    exit 1
fi