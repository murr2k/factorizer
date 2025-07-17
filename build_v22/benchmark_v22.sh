#!/bin/bash
# Benchmark script for CUDA Factorizer v2.2.0

echo "CUDA Factorizer v2.2.0 - Benchmark"
echo "=================================="
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Test different number sizes
test_numbers=(
    "1234567"           # 7 digits
    "123456789"         # 9 digits
    "12345678901234"    # 14 digits
    "1234567890123456"  # 16 digits
    "123456789012345678" # 18 digits
    "15482526220500967432610341" # 26 digits - the challenge
)

echo "Number,Digits,Time(s),Status" > benchmark_results.csv

for num in "${test_numbers[@]}"; do
    digits=${#num}
    echo "Testing $digits-digit number: $num"
    
    start_time=$(date +%s.%N)
    if "$SCRIPT_DIR/factorizer_v22" "$num" -q > /tmp/factorizer_output.txt 2>&1; then
        status="SUCCESS"
    else
        status="FAILED"
    fi
    end_time=$(date +%s.%N)
    
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    echo "$num,$digits,$elapsed,$status" >> benchmark_results.csv
    printf "  Time: %.3f seconds - %s\n" $elapsed $status
done

echo
echo "Results saved to benchmark_results.csv"
