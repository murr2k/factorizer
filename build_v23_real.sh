#!/bin/bash

# Build script for CUDA Factorizer v2.3.0 - Real Algorithm Edition
# This version uses ACTUAL ECM and QS algorithms instead of lookup tables

set -e

echo "Building CUDA Factorizer v2.3.0 - Real Algorithm Edition..."

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Create build directory
mkdir -p build_v23_real
cd build_v23_real

# Build the main factorizer
echo "Compiling main factorizer..."
nvcc -std=c++14 -O3 -arch=sm_75 -I.. \
    -o factorizer_v23_real \
    ../factorizer_v23_real.cu \
    -lcudart -lcurand

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "✓ Main factorizer compiled successfully"
else
    echo "✗ Main factorizer compilation failed"
    exit 1
fi

# Create test runner script
cat > run_v23_real.sh << 'EOF'
#!/bin/bash

# Test runner for CUDA Factorizer v2.3.0 Real

echo "Testing CUDA Factorizer v2.3.0 - Real Algorithm Edition"
echo "========================================================"

# Test cases from previous versions
echo
echo "Testing known factorization cases..."

# Small number test
echo "1. Testing small number (123456789):"
./factorizer_v23_real 123456789

# 26-digit test case
echo
echo "2. Testing 26-digit case (ECM should be selected):"
./factorizer_v23_real 15482526220500967432610341

# 86-bit test cases
echo
echo "3. Testing 86-bit case 1 (QS should be selected):"
./factorizer_v23_real 46095142970451885947574139

echo
echo "4. Testing 86-bit case 2:"
./factorizer_v23_real 71074534431598456802573371

echo
echo "5. Testing 86-bit case 3:"
./factorizer_v23_real 46394523650818021086494267

echo
echo "All tests completed!"
EOF

chmod +x run_v23_real.sh

# Create comprehensive test suite
cat > test_v23_comprehensive.sh << 'EOF'
#!/bin/bash

# Comprehensive test suite for v2.3.0 Real

echo "CUDA Factorizer v2.3.0 - Real Algorithm Comprehensive Test Suite"
echo "=================================================================="

# Test different algorithm paths
echo
echo "Testing algorithm selection logic..."

# Small numbers (should use trial division)
echo "Small numbers (trial division):"
./factorizer_v23_real 1001
./factorizer_v23_real 9999

# Medium numbers (should use Pollard's Rho)
echo
echo "Medium numbers (Pollard's Rho):"
./factorizer_v23_real 123456789123
./factorizer_v23_real 987654321987

# Large numbers (should use ECM)
echo
echo "Large numbers (ECM):"
./factorizer_v23_real 1048576000000000000000000000
./factorizer_v23_real 15482526220500967432610341

# Very large numbers (should use QS)
echo
echo "Very large numbers (QS):"
./factorizer_v23_real 46095142970451885947574139
./factorizer_v23_real 71074534431598456802573371

echo
echo "Comprehensive testing completed!"
EOF

chmod +x test_v23_comprehensive.sh

# Create performance benchmark
cat > benchmark_v23.sh << 'EOF'
#!/bin/bash

# Performance benchmark for v2.3.0 Real

echo "CUDA Factorizer v2.3.0 - Real Algorithm Performance Benchmark"
echo "=============================================================="

# Test performance with timing
echo
echo "Benchmarking different number types..."

# Function to time execution
time_execution() {
    local number=$1
    local description=$2
    echo "Testing $description: $number"
    start_time=$(date +%s.%N)
    ./factorizer_v23_real $number
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    echo "Elapsed time: ${elapsed}s"
    echo "----------------------------------------"
}

# Benchmark different cases
time_execution "123456789" "Small number"
time_execution "123456789123456789" "Medium number"
time_execution "15482526220500967432610341" "26-digit ECM case"
time_execution "46095142970451885947574139" "86-bit QS case"

echo
echo "Performance benchmark completed!"
EOF

chmod +x benchmark_v23.sh

# Create algorithm comparison test
cat > test_algorithm_comparison.sh << 'EOF'
#!/bin/bash

# Algorithm comparison test for v2.3.0

echo "Algorithm Comparison Test"
echo "========================="

# Test the same number with different algorithms by size
echo
echo "Testing algorithm selection on same number types..."

# We'll test numbers that are on the boundary between algorithms
echo "Testing 64-bit numbers (Pollard's Rho vs ECM boundary):"
./factorizer_v23_real 18446744073709551557  # Large 64-bit prime

echo
echo "Testing 84-bit numbers (ECM optimal range):"
./factorizer_v23_real 15482526220500967432610341

echo
echo "Testing 86-bit numbers (QS optimal range):"
./factorizer_v23_real 46095142970451885947574139

echo
echo "Algorithm comparison completed!"
EOF

chmod +x test_algorithm_comparison.sh

# Create validation script
cat > validate_v23.sh << 'EOF'
#!/bin/bash

# Validation script for v2.3.0 Real

echo "Validating CUDA Factorizer v2.3.0 - Real Algorithm Edition"
echo "==========================================================="

# Check if executable exists
if [ ! -f "./factorizer_v23_real" ]; then
    echo "Error: factorizer_v23_real not found"
    exit 1
fi

# Check CUDA device
echo "Checking CUDA device..."
nvidia-smi -L | head -1

# Test basic functionality
echo
echo "Testing basic functionality..."
./factorizer_v23_real 15 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Basic functionality test passed"
else
    echo "✗ Basic functionality test failed"
    exit 1
fi

# Test known factorizations
echo
echo "Testing known factorizations..."

# Test 21 = 3 × 7
result=$(./factorizer_v23_real 21 2>&1)
if echo "$result" | grep -q "3\|7"; then
    echo "✓ Small factorization test passed"
else
    echo "✗ Small factorization test failed"
    exit 1
fi

echo
echo "✓ All validation tests passed!"
echo "The real algorithm edition is ready for use!"
EOF

chmod +x validate_v23.sh

# Go back to main directory
cd ..

echo
echo "✓ Build completed successfully!"
echo "Build artifacts created in: build_v23_real/"
echo
echo "Available scripts:"
echo "  ./build_v23_real/run_v23_real.sh           - Basic test runner"
echo "  ./build_v23_real/test_v23_comprehensive.sh - Comprehensive tests"
echo "  ./build_v23_real/benchmark_v23.sh          - Performance benchmark"
echo "  ./build_v23_real/test_algorithm_comparison.sh - Algorithm comparison"
echo "  ./build_v23_real/validate_v23.sh           - Validation tests"
echo
echo "To run the factorizer:"
echo "  ./build_v23_real/factorizer_v23_real <number>"
echo
echo "Example:"
echo "  ./build_v23_real/factorizer_v23_real 15482526220500967432610341"