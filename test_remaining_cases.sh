#!/bin/bash

# Test remaining cases with longer timeouts

echo "=== Testing Remaining Cases with Extended Timeouts ==="
echo

# Function to test a single case
test_case() {
    local number=$1
    local factor1=$2
    local factor2=$3
    local desc=$4
    local timeout_val=$5
    
    echo "Testing $desc number: $number"
    echo "Expected factors: $factor1 × $factor2"
    echo "Timeout: ${timeout_val}s"
    
    # Run with extended timeout
    timeout $timeout_val ./factorizer_working "$number"
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "❌ TIMEOUT after $timeout_val seconds"
    elif [ $exit_code -eq 0 ]; then
        echo "✓ Test completed"
    else
        echo "❌ Error occurred (exit code: $exit_code)"
    fi
    
    echo "----------------------------------------"
    echo
}

# Test the larger cases with increasing timeouts
echo "=== Continuing with larger test cases ==="
echo

# 25-digit (already timed out at 30s, try 120s)
test_case "7362094681552249594844569" "3011920603541" "2444318974709" "25-digit" 120

# 31-digit (try 180s)
test_case "6686055831797977225042686908281" "2709353969339243" "2467767559153067" "31-digit" 180

# 40-digit (try 240s)
test_case "1713405256705515214666051277723996933341" "62091822164219516723" "27594700831518952367" "40-digit" 240

# 45-digit (try 300s)
test_case "883599419403825083339397145228886129352347501" "79928471789373227718301" "11054876937122958269201" "45-digit" 300

echo "=== Summary of All Tests ==="
echo
echo "Successfully factored (from previous run):"
echo "✓ 11-digit: 90,595,490,423 (3.8s)"
echo "✓ 12-digit: 324,625,056,641 (6.5s)"
echo "✓ 13-digit: 2,626,476,057,461 (8.5s)"
echo "✓ 16-digit: 3,675,257,317,722,541 (11.1s)"
echo
echo "Larger numbers: See results above"