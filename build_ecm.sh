#!/bin/bash
# Build script for ECM factorizer

echo "Building CUDA Factorizer with ECM support..."
echo "=========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA nvcc compiler not found!"
    exit 1
fi

# Display CUDA version
echo "CUDA Version:"
nvcc --version | head -1
echo

# Clean previous builds
echo "Cleaning previous builds..."
make -f Makefile.ecm clean

# Build main factorizer with ECM
echo "Building factorizer with ECM..."
make -f Makefile.ecm factorizer_ecm

if [ $? -eq 0 ]; then
    echo "✓ Main factorizer built successfully"
else
    echo "✗ Failed to build main factorizer"
    exit 1
fi

# Build test program
echo
echo "Building ECM test program..."
make -f Makefile.ecm test_ecm

if [ $? -eq 0 ]; then
    echo "✓ Test program built successfully"
else
    echo "✗ Failed to build test program"
    exit 1
fi

echo
echo "Build complete! Executables created:"
echo "  - factorizer_ecm    : Main factorizer with ECM support"
echo "  - test_ecm          : ECM algorithm test suite"
echo
echo "Usage examples:"
echo "  ./factorizer_ecm 1099511627791"
echo "  ./factorizer_ecm -a ecm 123456789012345678901"
echo "  ./factorizer_ecm --benchmark"
echo "  ./test_ecm"