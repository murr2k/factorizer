#!/bin/bash
# Run comprehensive test suite

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

echo "Running improved factorizer test suite..."
./test_improved

echo -e "\nTesting factorization on validated test cases..."

# Test the 8 validated cases
TEST_CASES=(
    "90595490423"
    "324625056641"
    "2626476057461"
    "3675257317722541"
    "7362094681552249594844569"
    "6686055831797977225042686908281"
    "1713405256705515214666051277723996933341"
    "883599419403825083339397145228886129352347501"
)

for number in "${TEST_CASES[@]}"; do
    echo -e "\nFactoring: $number"
    timeout 10 ./factorizer_improved "$number"
    if [ $? -eq 124 ]; then
        echo "Timeout after 10 seconds"
    fi
done
