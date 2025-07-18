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
