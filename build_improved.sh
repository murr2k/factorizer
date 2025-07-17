#!/bin/bash

# Build script for improved 128-bit factorizer

echo "Building improved 128-bit factorizer with hive-mind optimizations..."

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

# Create build directory
mkdir -p build_improved

# Compile flags
NVCC_FLAGS="-std=c++11 -arch=sm_75 -O3"
NVCC_FLAGS="$NVCC_FLAGS -Xcompiler -Wall"
NVCC_FLAGS="$NVCC_FLAGS -lcurand"

# Build main factorizer
echo "Compiling factorizer_cuda_128_improved_v2..."
nvcc $NVCC_FLAGS factorizer_cuda_128_improved_v2.cu -o build_improved/factorizer_improved

if [ $? -eq 0 ]; then
    echo "✓ Main factorizer built successfully"
else
    echo "✗ Failed to build main factorizer"
    exit 1
fi

# Build test suite
echo "Compiling test suite..."
nvcc $NVCC_FLAGS test_improved_factorizer.cu -o build_improved/test_improved

if [ $? -eq 0 ]; then
    echo "✓ Test suite built successfully"
else
    echo "✗ Failed to build test suite"
    exit 1
fi

# Create run script
cat > build_improved/run_factorizer.sh << 'EOF'
#!/bin/bash
# Wrapper script for running improved factorizer

# Set library paths for WSL2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Run with arguments
exec ./factorizer_improved "$@"
EOF

chmod +x build_improved/run_factorizer.sh

# Create test script
cat > build_improved/run_tests.sh << 'EOF'
#!/bin/bash
# Run comprehensive test suite

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

echo "Running improved factorizer test suite..."
./test_improved

echo -e "\nTesting factorization on validated test cases..."

# Test the 8 validated cases
TEST_CASES=(
    "90595490423"
    "324625056641"
    "2626476057461"
    "3675257317722541"
    "7362094681552249594844569"
    "6686055831797977225042686908281"
    "1713405256705515214666051277723996933341"
    "883599419403825083339397145228886129352347501"
)

for number in "${TEST_CASES[@]}"; do
    echo -e "\nFactoring: $number"
    timeout 10 ./factorizer_improved "$number"
    if [ $? -eq 124 ]; then
        echo "Timeout after 10 seconds"
    fi
done
EOF

chmod +x build_improved/run_tests.sh

echo -e "\n✓ Build complete!"
echo -e "\nTo run tests:"
echo "  cd build_improved && ./run_tests.sh"
echo -e "\nTo factor a number:"
echo "  cd build_improved && ./run_factorizer.sh <number>"
echo -e "\nExample:"
echo "  cd build_improved && ./run_factorizer.sh 90595490423"