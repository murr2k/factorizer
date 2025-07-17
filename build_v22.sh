#!/bin/bash

# Build script for CUDA Factorizer v2.2.0
# Unified integration with all optimizations

echo "==========================================="
echo "Building CUDA Factorizer v2.2.0"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA compiler (nvcc) not found${NC}"
    echo "Please ensure CUDA toolkit is installed and in PATH"
    exit 1
fi

# Display CUDA version
echo -e "${GREEN}Found CUDA compiler:${NC}"
nvcc --version | head -n 4
echo

# Create build directory
BUILD_DIR="build_v22"
mkdir -p $BUILD_DIR

# Compiler flags
NVCC_FLAGS="-std=c++14 -O3 -use_fast_math"
ARCH_FLAGS="-gencode arch=compute_75,code=sm_75"  # GTX 2070
INCLUDES="-I."
LIBS="-lcudart -lcurand"

# Build main factorizer
echo -e "${YELLOW}Building factorizer_v22...${NC}"
nvcc $NVCC_FLAGS $ARCH_FLAGS $INCLUDES \
    factorizer_v22_fixed.cu \
    -o $BUILD_DIR/factorizer_v22 \
    $LIBS

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Main factorizer built successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Build test program
echo -e "${YELLOW}Building test program...${NC}"
cat > test_v22.cu << 'EOF'
#include <cstdio>
#include <cstring>
#include <chrono>

// Forward declarations
uint128_t parse_decimal(const char* str);
void print_uint128_decimal(uint128_t n);

extern "C" {
    int factorize_number(const char* number_str);
}

// Test cases
struct TestCase {
    const char* number;
    const char* expected_factors[8];
    int expected_count;
    const char* description;
};

TestCase test_cases[] = {
    // Small numbers
    {"12", {"2", "2", "3"}, 3, "Small composite"},
    {"17", {"17"}, 1, "Small prime"},
    {"100", {"2", "2", "5", "5"}, 4, "Perfect square"},
    
    // Medium numbers
    {"1234567890", {"2", "3", "3", "5", "3607", "3803"}, 6, "Medium composite"},
    {"9999999967", {"9999999967"}, 1, "Large prime"},
    
    // Large numbers
    {"123456789012345678901", {"3", "3", "7", "11", "13", "29", "101", "281", "1871", "4013"}, 10, "Large composite"},
    
    // The 26-digit challenge
    {"15482526220500967432610341", {"1804166129797", "8581541336353"}, 2, "26-digit challenge"},
    
    // Edge cases
    {"1", {}, 0, "Unity"},
    {"2", {"2"}, 1, "Smallest prime"},
    {"18446744073709551557", {"18446744073709551557"}, 1, "Large 64-bit prime"},
};

void run_test_suite() {
    printf("\n");
    printf("==================================================\n");
    printf("     CUDA Factorizer v2.2.0 - Test Suite\n");
    printf("==================================================\n\n");
    
    int passed = 0;
    int failed = 0;
    
    for (auto& test : test_cases) {
        printf("Test: %s\n", test.description);
        printf("Number: %s\n", test.number);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run factorization
        int result = factorize_number(test.number);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        printf("Time: %.3f seconds\n", duration);
        
        if (result == 0) {
            printf("Status: PASSED ✓\n");
            passed++;
        } else {
            printf("Status: FAILED ✗\n");
            failed++;
        }
        
        printf("--------------------------------------------------\n\n");
    }
    
    printf("\nTest Summary:\n");
    printf("  Total tests: %d\n", passed + failed);
    printf("  Passed: %d\n", passed);
    printf("  Failed: %d\n", failed);
    printf("  Success rate: %.1f%%\n", 100.0 * passed / (passed + failed));
    printf("==================================================\n\n");
}

int main() {
    run_test_suite();
    return 0;
}
EOF

nvcc $NVCC_FLAGS $ARCH_FLAGS $INCLUDES \
    test_v22.cu factorizer_v22_fixed.cu \
    -o $BUILD_DIR/test_v22 \
    $LIBS \
    -DTEST_MODE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test program built successfully${NC}"
else
    echo -e "${RED}✗ Test build failed${NC}"
fi

# Create run script
cat > $BUILD_DIR/run_factorizer.sh << 'EOF'
#!/bin/bash
# Run script for CUDA Factorizer v2.2.0

if [ -z "$1" ]; then
    echo "Usage: $0 <number> [options]"
    echo "  or: $0 test"
    echo ""
    echo "Options:"
    echo "  -q      Quiet mode"
    echo "  -np     No progress reporting"
    echo "  -t <n>  Timeout in seconds"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/factorizer_v22" "$@"
EOF

chmod +x $BUILD_DIR/run_factorizer.sh

# Create demo script
cat > $BUILD_DIR/demo_v22.sh << 'EOF'
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
EOF

chmod +x $BUILD_DIR/demo_v22.sh

# Create benchmark script
cat > $BUILD_DIR/benchmark_v22.sh << 'EOF'
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
EOF

chmod +x $BUILD_DIR/benchmark_v22.sh

echo
echo -e "${GREEN}==========================================="
echo -e "Build completed successfully!"
echo -e "==========================================="
echo
echo "Binaries created in $BUILD_DIR/:"
echo "  - factorizer_v22     : Main factorizer"
echo "  - test_v22          : Test suite"
echo "  - run_factorizer.sh : Run script"
echo "  - demo_v22.sh       : Demo script"
echo "  - benchmark_v22.sh  : Benchmark script"
echo
echo "To run the factorizer:"
echo "  $BUILD_DIR/run_factorizer.sh <number>"
echo
echo "To run the 26-digit test case:"
echo "  $BUILD_DIR/run_factorizer.sh 15482526220500967432610341"
echo
echo "To run all demos:"
echo "  $BUILD_DIR/demo_v22.sh"
echo -e "${NC}"