#!/bin/bash

# Test script for new 128-bit factorization test cases

echo "================================================"
echo "Testing New 128-bit Factorization Test Cases"
echo "================================================"

# Test cases array (number, factor1, factor2, description)
declare -a test_cases=(
    "90595490423:428759:211297:11-digit semiprime"
    "324625056641:408337:794993:12-digit semiprime"
    "2626476057461:1321171:1987991:13-digit semiprime"
    "3675257317722541:91709393:40075037:16-digit semiprime"
    "7362094681552249594844569:3011920603541:2444318974709:25-digit semiprime"
    "6686055831797977225042686908281:2709353969339243:2467767559153067:31-digit semiprime"
    "1713405256705515214666051277723996933341:62091822164219516723:27594700831518952367:40-digit semiprime"
    "883599419403825083339397145228886129352347501:79928471789373227718301:11054876937122958269201:45-digit semiprime"
)

# Function to test factorization
test_factorization() {
    local number=$1
    local expected_f1=$2
    local expected_f2=$3
    local description=$4
    local utility=$5
    
    echo -e "\n[TEST] $description"
    echo "Number: $number"
    echo "Expected: $expected_f1 × $expected_f2"
    
    # Test with timeout
    echo -n "Testing with $utility... "
    
    # Create a temporary file for output
    tmpfile=$(mktemp)
    
    # Run with timeout and capture output
    timeout 30s ./$utility $number > $tmpfile 2>&1
    result=$?
    
    if [ $result -eq 124 ]; then
        echo "TIMEOUT (>30s)"
    elif [ $result -eq 0 ]; then
        # Check if factors were found
        if grep -q "Factor" $tmpfile; then
            echo "SUCCESS"
            grep "Factor" $tmpfile | head -2
            
            # Try to extract and verify factors
            factors=$(grep -oE "Factor found: [0-9]+" $tmpfile | grep -oE "[0-9]+")
            if [ ! -z "$factors" ]; then
                echo "Verifying factors..."
                python3 -c "
import sys
factors = '$factors'.split()
if len(factors) >= 2:
    f1, f2 = int(factors[0]), int(factors[1])
    product = f1 * f2
    if product == $number:
        print('✓ Verification successful')
    else:
        print('✗ Verification failed')
"
            fi
        else
            echo "NO FACTORS FOUND"
            tail -5 $tmpfile
        fi
    else
        echo "ERROR (exit code: $result)"
        tail -5 $tmpfile
    fi
    
    rm -f $tmpfile
}

# Test each case with both utilities
for test_case in "${test_cases[@]}"; do
    IFS=':' read -r number f1 f2 desc <<< "$test_case"
    
    # Determine which utility to use based on size
    digits=${#number}
    
    if [ $digits -le 20 ]; then
        # Use regular factorizer for smaller numbers
        test_factorization "$number" "$f1" "$f2" "$desc" "factorizer"
    else
        # Use 128-bit factorizer for larger numbers
        test_factorization "$number" "$f1" "$f2" "$desc" "factorizer128"
    fi
done

echo -e "\n================================================"
echo "Performance Summary"
echo "================================================"

# Quick performance test on subset
echo -e "\nTiming tests (with 5s timeout):"
test_numbers=(90595490423 2626476057461 7362094681552249594844569)

for num in "${test_numbers[@]}"; do
    echo -n "Number (${#num} digits): "
    start_time=$(date +%s.%N)
    timeout 5s ./factorizer $num > /dev/null 2>&1
    result=$?
    end_time=$(date +%s.%N)
    
    if [ $result -eq 124 ]; then
        echo "TIMEOUT"
    else
        runtime=$(echo "$end_time - $start_time" | bc)
        echo "$runtime seconds"
    fi
done

echo -e "\n================================================"
echo "Creating Updated Test Cases File"
echo "================================================"

# Generate updated test_128bit.cu snippet
cat > updated_test_cases.cpp << 'EOF'
// Updated test cases for test_128bit.cu
void initializeTestCases() {
    test_cases = {
        // Small semiprimes (verifiable)
        {"90595490423", "428759", "211297", "11-digit semiprime"},
        {"324625056641", "408337", "794993", "12-digit semiprime"},
        {"2626476057461", "1321171", "1987991", "13-digit semiprime"},
        
        // Medium semiprimes
        {"3675257317722541", "91709393", "40075037", "16-digit semiprime"},
        {"7362094681552249594844569", "3011920603541", "2444318974709", "25-digit semiprime"},
        
        // Large semiprimes
        {"6686055831797977225042686908281", "2709353969339243", "2467767559153067", "31-digit semiprime"},
        {"1713405256705515214666051277723996933341", "62091822164219516723", "27594700831518952367", "40-digit semiprime"},
        
        // Extra large (45-digit)
        {"883599419403825083339397145228886129352347501", "79928471789373227718301", "11054876937122958269201", "45-digit semiprime"},
        
        // Edge cases (keep existing valid ones)
        {"4", "2", "2", "Smallest semiprime"},
        {"6", "2", "3", "Product of first two primes"},
        {"9", "3", "3", "Square of prime"},
    };
}
EOF

echo "Updated test cases saved to: updated_test_cases.cpp"

echo -e "\n✅ Test suite completed!"