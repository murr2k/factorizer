#!/bin/bash
# Demo script for CUDA Factorizer v2.2.0

echo "CUDA Factorizer v2.2.0 - Demo"
echo "=============================="
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Demo 1: Small number
echo "Demo 1: Small number (12-digit)"
echo "Number: 123456789012"
"$SCRIPT_DIR/factorizer_v22" 123456789012
echo

# Demo 2: Medium number
echo "Demo 2: Medium number (15-digit)"
echo "Number: 123456789012345"
"$SCRIPT_DIR/factorizer_v22" 123456789012345
echo

# Demo 3: The 26-digit challenge
echo "Demo 3: The 26-digit challenge"
echo "Number: 15482526220500967432610341"
echo "Expected: 1804166129797 × 8581541336353"
"$SCRIPT_DIR/factorizer_v22" 15482526220500967432610341
echo

# Demo 4: Test mode
echo "Demo 4: Running built-in test"
"$SCRIPT_DIR/factorizer_v22" test

echo "Demo complete!"
