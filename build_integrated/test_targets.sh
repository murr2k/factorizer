#!/bin/bash
# Test both target cases for CUDA Factorizer v2.2.0

echo "CUDA Factorizer v2.2.0 - Target Case Tests"
echo "==========================================="
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Test 26-digit case (should use ECM)
echo "Test 1: 26-digit case (ECM optimal)"
echo "Number: 15482526220500967432610341"
echo "Expected: ECM with factors 1804166129797 × 8581541336353"
echo
"$SCRIPT_DIR/factorizer_integrated" test_26digit
echo

# Test 86-bit case (should use QS)
echo "Test 2: 86-bit case (QS optimal)"
echo "Number: 71123818302723020625487649"
echo "Expected: QS with factors 7574960675251 × 9389331687899"
echo
"$SCRIPT_DIR/factorizer_integrated" test_86bit
echo

echo "Target case tests completed!"
