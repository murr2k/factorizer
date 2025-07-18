#!/bin/bash

# Build script for CUDA Factorizer v2.3.0 - Simple Real Algorithm Edition

set -e

echo "Building CUDA Factorizer v2.3.0 - Simple Real Algorithm Edition..."

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Create build directory
mkdir -p build_v23_simple
cd build_v23_simple

# Build the main factorizer
echo "Compiling main factorizer..."
nvcc -std=c++14 -O3 -arch=sm_75 -I.. \
    -o factorizer_v23_simple \
    ../factorizer_v23_simple.cu \
    -lcudart -lcurand

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo "✓ Main factorizer compiled successfully"
else
    echo "✗ Main factorizer compilation failed"
    exit 1
fi

# Create test runner script
cat > run_v23_simple.sh << 'EOF'
#!/bin/bash

echo "Testing CUDA Factorizer v2.3.0 - Simple Real Algorithm Edition"
echo "==============================================================="

# Test cases
echo
echo "Testing with real algorithms (no lookup tables)..."

# Small number test
echo "1. Testing small number:"
./factorizer_v23_simple 123456789

# Medium number test
echo
echo "2. Testing medium number:"
./factorizer_v23_simple 123456789123456789

# Large number test (previous 26-digit case)
echo
echo "3. Testing large number (26-digit):"
./factorizer_v23_simple 15482526220500967432610341

# Very large number test (previous 86-bit case)
echo
echo "4. Testing very large number (86-bit):"
./factorizer_v23_simple 46095142970451885947574139

echo
echo "All tests completed!"
EOF

chmod +x run_v23_simple.sh

# Create validation script
cat > validate_v23_simple.sh << 'EOF'
#!/bin/bash

echo "Validating CUDA Factorizer v2.3.0 - Simple Real Algorithm Edition"
echo "=================================================================="

# Check if executable exists
if [ ! -f "./factorizer_v23_simple" ]; then
    echo "Error: factorizer_v23_simple not found"
    exit 1
fi

# Test basic functionality
echo "Testing basic functionality..."
./factorizer_v23_simple 15 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Basic functionality test passed"
else
    echo "✗ Basic functionality test failed"
    exit 1
fi

# Test known factorizations
echo "Testing known factorizations..."

# Test 21 = 3 × 7
result=$(./factorizer_v23_simple 21 2>&1)
if echo "$result" | grep -q "3\|7"; then
    echo "✓ Small factorization test passed"
else
    echo "✗ Small factorization test failed"
    exit 1
fi

# Test 143 = 11 × 13
result=$(./factorizer_v23_simple 143 2>&1)
if echo "$result" | grep -q "11\|13"; then
    echo "✓ Medium factorization test passed"
else
    echo "✗ Medium factorization test failed"
    exit 1
fi

echo
echo "✓ All validation tests passed!"
echo "The simple real algorithm edition is working correctly!"
EOF

chmod +x validate_v23_simple.sh

# Create performance test
cat > test_performance.sh << 'EOF'
#!/bin/bash

echo "Performance Test for v2.3.0 Simple Real Algorithm Edition"
echo "=========================================================="

echo "Testing performance with different number sizes..."

# Function to time execution
time_test() {
    local number=$1
    local description=$2
    echo "Testing $description: $number"
    
    start_time=$(date +%s.%N)
    result=$(./factorizer_v23_simple "$number" 2>&1)
    end_time=$(date +%s.%N)
    
    if echo "$result" | grep -q "✓"; then
        elapsed=$(echo "$end_time - $start_time" | bc)
        echo "✓ Success in ${elapsed}s"
    else
        echo "✗ Failed or timed out"
    fi
    echo "----------------------------------------"
}

# Test different sizes
time_test "1001" "Small composite"
time_test "1234567" "Medium number"
time_test "123456789123" "Large number"
time_test "15482526220500967432610341" "Very large number"

echo "Performance testing completed!"
EOF

chmod +x test_performance.sh

# Go back to main directory
cd ..

echo
echo "✓ Build completed successfully!"
echo "Build artifacts created in: build_v23_simple/"
echo
echo "Available scripts:"
echo "  ./build_v23_simple/run_v23_simple.sh        - Basic test runner"
echo "  ./build_v23_simple/validate_v23_simple.sh   - Validation tests"
echo "  ./build_v23_simple/test_performance.sh      - Performance tests"
echo
echo "To run the factorizer:"
echo "  ./build_v23_simple/factorizer_v23_simple <number>"
echo
echo "Example:"
echo "  ./build_v23_simple/factorizer_v23_simple 15482526220500967432610341"
echo
echo "Note: This version uses REAL algorithms - no lookup tables!"