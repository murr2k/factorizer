#!/bin/bash
# Build script for Quadratic Sieve implementation

echo "Building Quadratic Sieve implementation..."
echo "========================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    exit 1
fi

# Show CUDA version
echo "CUDA Version:"
nvcc --version | head -n 4
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
make -f Makefile.qs clean

# Build all targets
echo "Building all targets..."
make -f Makefile.qs all

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "Running basic tests..."
    echo "======================"
    
    # Run basic test
    if [ -f ./test_qs ]; then
        ./test_qs
    else
        echo "Error: test_qs executable not found!"
        exit 1
    fi
    
    echo ""
    echo "To run additional tests:"
    echo "  ./test_qs perf    - Performance comparison"
    echo "  ./test_qs gpu     - GPU kernel tests"
    echo ""
    echo "To integrate with main factorizer:"
    echo "  1. Include 'quadratic_sieve.cuh' in your main file"
    echo "  2. Link with quadratic_sieve_optimized.cu"
    echo "  3. Call QuadraticSieve class or quadratic_sieve_factor() function"
else
    echo "Build failed!"
    exit 1
fi