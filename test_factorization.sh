#!/bin/bash
# Test script for 128-bit factorization debugging

echo "=== 128-bit Factorization Test Suite ==="
echo "Building test programs..."

# Clean and build
make clean
make test_128bit_arithmetic
make factorizer_v21_128bit
make factorizer_v21_128bit_diagnostic

echo -e "\n=== Running arithmetic tests ==="
./test_128bit_arithmetic

echo -e "\n=== Testing with small known composites ==="
echo "Testing 299 = 13 × 23"
./factorizer_v21_128bit_diagnostic 299 1

echo -e "\nTesting 8633 = 89 × 97"
./factorizer_v21_128bit_diagnostic 8633 1

echo -e "\nTesting 1000000007 × 1000000009 = 1000000016000000063"
./factorizer_v21_128bit_diagnostic 1000000016000000063 1

echo -e "\n=== Testing with larger numbers ==="
echo "Testing 12345678901234567 × 98765432109876543 = 1219326312467611126347425615330881"
./factorizer_v21_128bit_diagnostic 1219326312467611126347425615330881 1

echo -e "\n=== Comparing original vs improved modmul ==="
echo "Testing with original modmul:"
./factorizer_v21_128bit_diagnostic 1000000016000000063 0

echo -e "\nTesting with improved modmul:"
./factorizer_v21_128bit_diagnostic 1000000016000000063 1

echo -e "\n=== Testing the original implementation ==="
./factorizer_v21_128bit 299
./factorizer_v21_128bit 1000000016000000063