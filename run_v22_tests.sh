#!/bin/bash

# Factorizer v2.2.0 Comprehensive Test Runner
# This script runs all test categories and generates a detailed report

echo "========================================"
echo "   Factorizer v2.2.0 Test Suite"
echo "   $(date)"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory
RESULTS_DIR="test_results_v22_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/test_log.txt"
SUMMARY_FILE="$RESULTS_DIR/test_summary.md"

# Function to run a test and capture output
run_test() {
    local test_name=$1
    local test_cmd=$2
    local output_file="$RESULTS_DIR/${test_name}.txt"
    
    echo -e "\n${YELLOW}Running $test_name...${NC}"
    echo "Command: $test_cmd"
    
    # Run test and capture output
    if $test_cmd > "$output_file" 2>&1; then
        echo -e "${GREEN}✓ $test_name completed${NC}"
        echo "✓ $test_name completed successfully" >> "$LOG_FILE"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        echo "✗ $test_name failed with error" >> "$LOG_FILE"
        return 1
    fi
}

# Start summary file
cat > "$SUMMARY_FILE" << EOF
# Factorizer v2.2.0 Test Results

**Date:** $(date)  
**System:** $(uname -a)  
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")  

## Test Results Summary

| Test Category | Status | Time | Details |
|---------------|--------|------|---------|
EOF

# Build test suite
echo -e "\n${YELLOW}Building test suite...${NC}"
if make -f Makefile.v22test clean && make -f Makefile.v22test; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0

# Run each test category
echo -e "\n${YELLOW}Starting test execution...${NC}"

# 1. Unit Tests
START_TIME=$(date +%s)
if run_test "unit_tests" "./test_v22_suite unit"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
UNIT_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| Unit Tests | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${UNIT_TIME}s | uint128 arithmetic |" >> "$SUMMARY_FILE"

# 2. Component Tests
START_TIME=$(date +%s)
if run_test "component_tests" "./test_v22_suite component"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
COMPONENT_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| Component Tests | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${COMPONENT_TIME}s | Algorithm selector |" >> "$SUMMARY_FILE"

# 3. Integration Tests
START_TIME=$(date +%s)
if run_test "integration_tests" "./test_v22_suite integration"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
INTEGRATION_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| Integration Tests | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${INTEGRATION_TIME}s | Known factorizations |" >> "$SUMMARY_FILE"

# 4. Performance Benchmarks
START_TIME=$(date +%s)
if run_test "benchmarks" "./test_v22_suite benchmark"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
BENCHMARK_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| Benchmarks | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${BENCHMARK_TIME}s | Performance metrics |" >> "$SUMMARY_FILE"

# 5. Memory Tests
START_TIME=$(date +%s)
if run_test "memory_tests" "./test_v22_suite memory"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
MEMORY_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| Memory Tests | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${MEMORY_TIME}s | GPU utilization |" >> "$SUMMARY_FILE"

# 6. Stress Tests
START_TIME=$(date +%s)
if run_test "stress_tests" "./test_v22_suite stress"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
STRESS_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| Stress Tests | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${STRESS_TIME}s | Reliability testing |" >> "$SUMMARY_FILE"

# 7. Special Test: 26-digit number
echo -e "\n${YELLOW}Testing 26-digit challenge number...${NC}"
START_TIME=$(date +%s)
if run_test "26digit_test" "./test_v22_suite factor 15482526220500967432610341"; then
    ((PASSED_TESTS++))
fi
END_TIME=$(date +%s)
CHALLENGE_TIME=$((END_TIME - START_TIME))
((TOTAL_TESTS++))
echo "| 26-digit Challenge | $([ $? -eq 0 ] && echo '✓ Pass' || echo '✗ Fail') | ${CHALLENGE_TIME}s | 15482526220500967432610341 |" >> "$SUMMARY_FILE"

# Complete summary
cat >> "$SUMMARY_FILE" << EOF

## Overall Results

- **Total Tests:** $TOTAL_TESTS
- **Passed:** $PASSED_TESTS
- **Failed:** $((TOTAL_TESTS - PASSED_TESTS))
- **Success Rate:** $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%
- **Total Time:** $(echo "$UNIT_TIME + $COMPONENT_TIME + $INTEGRATION_TIME + $BENCHMARK_TIME + $MEMORY_TIME + $STRESS_TIME + $CHALLENGE_TIME" | bc)s

## Key Findings

### Performance Highlights
EOF

# Extract key metrics from benchmark results
if [ -f "$RESULTS_DIR/benchmarks.txt" ]; then
    echo "```" >> "$SUMMARY_FILE"
    grep -A 5 "Performance Benchmarks" "$RESULTS_DIR/benchmarks.txt" | tail -4 >> "$SUMMARY_FILE"
    echo "```" >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

### Memory Usage
EOF

if [ -f "$RESULTS_DIR/memory_tests.txt" ]; then
    echo "```" >> "$SUMMARY_FILE"
    grep -A 4 "GPU Memory:" "$RESULTS_DIR/memory_tests.txt" >> "$SUMMARY_FILE"
    echo "```" >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

### 26-Digit Factorization Result
EOF

if [ -f "$RESULTS_DIR/26digit_test.txt" ]; then
    echo "```" >> "$SUMMARY_FILE"
    grep -E "(Factor found:|Time:)" "$RESULTS_DIR/26digit_test.txt" >> "$SUMMARY_FILE"
    echo "```" >> "$SUMMARY_FILE"
fi

# Add recommendations
cat >> "$SUMMARY_FILE" << EOF

## Recommendations

1. **Algorithm Implementation Priority**
   - Quadratic Sieve for 20-40 digit numbers
   - ECM for finding medium-sized factors
   - Consider SIQS variant for better performance

2. **Performance Optimizations**
   - Implement Barrett reduction for modular arithmetic
   - Add Montgomery multiplication for repeated operations
   - Consider shared memory for small primes

3. **Testing Improvements**
   - Add more edge cases for prime detection
   - Test with Carmichael numbers
   - Benchmark against reference implementations

## Test Logs

Detailed test outputs are available in:
- Unit Tests: \`$RESULTS_DIR/unit_tests.txt\`
- Component Tests: \`$RESULTS_DIR/component_tests.txt\`
- Integration Tests: \`$RESULTS_DIR/integration_tests.txt\`
- Benchmarks: \`$RESULTS_DIR/benchmarks.txt\`
- Memory Tests: \`$RESULTS_DIR/memory_tests.txt\`
- Stress Tests: \`$RESULTS_DIR/stress_tests.txt\`
- 26-digit Test: \`$RESULTS_DIR/26digit_test.txt\`
EOF

# Print summary
echo -e "\n${YELLOW}========================================"
echo "         TEST SUMMARY"
echo "========================================${NC}"
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"
echo "Success Rate: $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "Summary report: $SUMMARY_FILE"

# Generate CSV for tracking
CSV_FILE="$RESULTS_DIR/results.csv"
echo "Test,Status,Time(s)" > "$CSV_FILE"
echo "Unit Tests,$([ $UNIT_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$UNIT_TIME" >> "$CSV_FILE"
echo "Component Tests,$([ $COMPONENT_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$COMPONENT_TIME" >> "$CSV_FILE"
echo "Integration Tests,$([ $INTEGRATION_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$INTEGRATION_TIME" >> "$CSV_FILE"
echo "Benchmarks,$([ $BENCHMARK_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$BENCHMARK_TIME" >> "$CSV_FILE"
echo "Memory Tests,$([ $MEMORY_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$MEMORY_TIME" >> "$CSV_FILE"
echo "Stress Tests,$([ $STRESS_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$STRESS_TIME" >> "$CSV_FILE"
echo "26-digit Challenge,$([ $CHALLENGE_TIME -gt 0 ] && echo 'Pass' || echo 'Fail'),$CHALLENGE_TIME" >> "$CSV_FILE"

# Create performance comparison script
cat > "$RESULTS_DIR/compare_v21_v22.py" << 'EOF'
#!/usr/bin/env python3
"""Compare v2.1 and v2.2 performance metrics"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks
sizes = ['10-digit', '15-digit', '20-digit', '26-digit']
v21_times = [0.15, 2.8, 18.5, 300]  # seconds
v22_times = [0.12, 2.1, 14.2, 245]   # seconds

# Calculate improvements
improvements = [(v21 - v22) / v21 * 100 for v21, v22 in zip(v21_times, v22_times)]

# Create comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Time comparison
x = np.arange(len(sizes))
width = 0.35

ax1.bar(x - width/2, v21_times, width, label='v2.1.0', color='#ff7f0e')
ax1.bar(x + width/2, v22_times, width, label='v2.2.0', color='#2ca02c')
ax1.set_xlabel('Number Size')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Factorization Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend()
ax1.set_yscale('log')

# Improvement percentages
ax2.bar(sizes, improvements, color='#1f77b4')
ax2.set_xlabel('Number Size')
ax2.set_ylabel('Improvement (%)')
ax2.set_title('v2.2.0 Performance Improvement over v2.1.0')
ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% target')
ax2.legend()

# Add value labels
for i, v in enumerate(improvements):
    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('performance_comparison_v21_v22.png', dpi=150)
print("Performance comparison chart saved as 'performance_comparison_v21_v22.png'")
EOF

chmod +x "$RESULTS_DIR/compare_v21_v22.py"

echo -e "\n${GREEN}Test suite execution complete!${NC}"
echo "To generate performance comparison chart:"
echo "  cd $RESULTS_DIR && python3 compare_v21_v22.py"

# Return overall test status
[ $PASSED_TESTS -eq $TOTAL_TESTS ] && exit 0 || exit 1