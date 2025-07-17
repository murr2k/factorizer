#!/bin/bash

# Test all 8 validated test cases with the improved factorizer

echo "=== Testing Improved 128-bit Factorizer with Validated Test Cases ==="
echo

# Array of test cases
declare -a TEST_CASES=(
    "90595490423:428759:211297:11-digit"
    "324625056641:408337:794993:12-digit"
    "2626476057461:1321171:1987991:13-digit"
    "3675257317722541:91709393:40075037:16-digit"
    "7362094681552249594844569:3011920603541:2444318974709:25-digit"
    "6686055831797977225042686908281:2709353969339243:2467767559153067:31-digit"
    "1713405256705515214666051277723996933341:62091822164219516723:27594700831518952367:40-digit"
    "883599419403825083339397145228886129352347501:79928471789373227718301:11054876937122958269201:45-digit"
)

# Function to test a single case
test_case() {
    local number=$1
    local factor1=$2
    local factor2=$3
    local desc=$4
    
    echo "Testing $desc number: $number"
    echo "Expected factors: $factor1 × $factor2"
    
    # Run with timeout
    timeout 30 ./factorizer_working "$number"
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "❌ TIMEOUT after 30 seconds"
    elif [ $exit_code -eq 0 ]; then
        echo "✓ Test completed"
    else
        echo "❌ Error occurred (exit code: $exit_code)"
    fi
    
    echo "----------------------------------------"
    echo
}

# Run all tests
for test in "${TEST_CASES[@]}"; do
    IFS=':' read -r number factor1 factor2 desc <<< "$test"
    test_case "$number" "$factor1" "$factor2" "$desc"
done

echo "=== Test Summary ==="
echo "Test suite completed. Check results above for each test case."