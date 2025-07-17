#!/bin/bash
# Test different algorithms for CUDA Factorizer v2.2.0

echo "CUDA Factorizer v2.2.0 - Algorithm Tests"
echo "========================================="
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Test small number (should use trial division)
echo "Test 1: Small number (trial division)"
echo "Number: 123456789"
"$SCRIPT_DIR/factorizer_integrated" 123456789
echo

# Test medium number (should use Pollard's Rho)
echo "Test 2: Medium number (Pollard's Rho)"
echo "Number: 1234567890123"
"$SCRIPT_DIR/factorizer_integrated" 1234567890123
echo

# Test large prime (should try multiple algorithms)
echo "Test 3: Large prime (multiple algorithms)"
echo "Number: 1000000000000000003"
"$SCRIPT_DIR/factorizer_integrated" 1000000000000000003
echo

echo "Algorithm tests completed!"
