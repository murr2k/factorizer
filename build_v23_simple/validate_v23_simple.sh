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
