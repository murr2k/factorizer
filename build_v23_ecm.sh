#!/bin/bash

# Build script for CUDA Factorizer v2.3.1 - ECM Edition

set -e

echo "Building CUDA Factorizer v2.3.1 - ECM Edition..."

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Create build directory
mkdir -p build_v23_ecm

# Build the ECM factorizer
echo "Compiling ECM factorizer..."
nvcc -std=c++14 -O3 -arch=sm_75 -I. \
    -o build_v23_ecm/factorizer_v23_ecm \
    factorizer_v23_ecm.cu \
    -lcudart -lcurand

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "✓ ECM factorizer compiled successfully"
else
    echo "✗ ECM factorizer compilation failed"
    exit 1
fi

echo
echo "✓ Build completed successfully!"
echo
echo "To run the ECM factorizer:"
echo "  ./build_v23_ecm/factorizer_v23_ecm <number>"
echo
echo "Example:"
echo "  ./build_v23_ecm/factorizer_v23_ecm 139789207152250802634791"