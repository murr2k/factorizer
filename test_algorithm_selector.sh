#!/bin/bash
# Test script for algorithm selector

echo "===================================="
echo "Algorithm Selector Test Suite"
echo "===================================="
echo

# Function to test a number
test_number() {
    local number=$1
    local description=$2
    
    echo "----------------------------------------"
    echo "Test: $description"
    echo "Number: $number"
    echo "----------------------------------------"
    ./algorithm_selector "$number"
    echo
    echo
}

# Build the algorithm selector
echo "Building algorithm selector..."
make -f Makefile.intelligent algorithm_selector
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi
echo

# Test cases

# 1. Small prime
test_number "97" "Small prime (2 digits)"

# 2. Small composite
test_number "143" "Small composite (11 × 13)"

# 3. Perfect power
test_number "1024" "Perfect power (2^10)"

# 4. Perfect square
test_number "625" "Perfect square (25^2)"

# 5. Smooth number
test_number "30030" "Smooth number (2×3×5×7×11×13)"

# 6. Medium prime
test_number "1000000007" "Medium prime (10 digits)"

# 7. Medium composite
test_number "1234567890123" "Medium composite"

# 8. Large semiprime
test_number "15482526220500967432610341" "Large semiprime (26 digits)"

# 9. Mersenne number
test_number "2147483647" "Mersenne number (2^31 - 1)"

# 10. Product of close primes
test_number "299060819409" "Product of close primes"

# 11. RSA-like number (product of two large primes)
test_number "3233" "Small RSA-like (61 × 53)"

# 12. Number with many small factors
test_number "362880" "9! (many small factors)"

echo "===================================="
echo "Test suite completed"
echo "===================================="