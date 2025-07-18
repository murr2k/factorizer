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
