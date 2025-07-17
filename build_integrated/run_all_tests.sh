#!/bin/bash
# Comprehensive test suite for CUDA Factorizer v2.2.0

echo "CUDA Factorizer v2.2.0 - Comprehensive Test Suite"
echo "================================================="
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run validation first
echo "Step 1: Validation Test"
echo "----------------------"
"$SCRIPT_DIR/validate_integrated"
if [ $? -ne 0 ]; then
    echo "Validation failed - aborting tests"
    exit 1
fi
echo

# Test target cases
echo "Step 2: Target Case Tests"
echo "-------------------------"
"$SCRIPT_DIR/test_targets.sh"
echo

# Test algorithms
echo "Step 3: Algorithm Tests"
echo "----------------------"
"$SCRIPT_DIR/test_algorithms.sh"
echo

echo "================================================="
echo "All tests completed!"
echo "================================================="
