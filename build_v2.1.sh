#!/bin/bash
# Build script for CUDA Factorizer v2.1.0

echo "========================================="
echo "Building CUDA Factorizer v2.1.0"
echo "========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please ensure CUDA Toolkit is installed and in PATH"
    exit 1
fi

# Display CUDA version
echo "CUDA Version:"
nvcc --version | grep release
echo

# Clean previous builds
echo "Cleaning previous builds..."
make -f Makefile.v2.1 clean

# Build main program and tests
echo "Building factorizer v2.1..."
if make -f Makefile.v2.1 all; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo
echo "========================================="
echo "Build artifacts created:"
echo "========================================="
ls -la factorizer_v2.1 test_v2.1 2>/dev/null

echo
echo "========================================="
echo "Quick test commands:"
echo "========================================="
echo "1. Run tests:        ./test_v2.1"
echo "2. Run benchmark:    ./factorizer_v2.1 --benchmark"
echo "3. Factor a number:  ./factorizer_v2.1 90595490423"
echo "4. Show help:        ./factorizer_v2.1 --help"
echo

# Optional: Run quick test
read -p "Run quick test now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running quick factorization test..."
    ./factorizer_v2.1 --no-progress 90595490423
fi

echo
echo "Build complete! 🚀"