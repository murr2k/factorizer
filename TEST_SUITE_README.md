# Factorizer v2.2.0 Test Suite

## Overview

This comprehensive test suite validates the functionality, performance, and reliability of the Factorizer v2.2.0 implementation. It includes unit tests, component tests, integration tests, benchmarks, and stress tests.

## Files

- `test_v22_suite.cu` - Main test suite implementation
- `benchmark_results_v22.md` - Expected performance metrics and analysis
- `Makefile.v22test` - Build configuration for the test suite
- `run_v22_tests.sh` - Automated test runner script
- `validate_26digit.cu` - Specific validation for the 26-digit challenge number
- `TEST_SUITE_README.md` - This file

## Quick Start

### Build the test suite:
```bash
make -f Makefile.v22test
```

### Run all tests:
```bash
./run_v22_tests.sh
```

### Run specific test categories:
```bash
# Unit tests only
./test_v22_suite unit

# Component tests
./test_v22_suite component

# Integration tests
./test_v22_suite integration

# Performance benchmarks
./test_v22_suite benchmark

# Memory and GPU utilization tests
./test_v22_suite memory

# Stress tests
./test_v22_suite stress

# Test specific number
./test_v22_suite factor 15482526220500967432610341
```

## Test Categories

### 1. Unit Tests
Tests fundamental uint128_t arithmetic operations:
- Addition with carry propagation
- Subtraction with borrow
- Multiplication with overflow handling
- Shift operations
- GCD calculation

### 2. Component Tests
Tests individual algorithm components:
- Pollard's Rho implementation
- Algorithm selector logic
- Modular arithmetic operations
- Random number generation

### 3. Integration Tests
Tests complete factorization workflow with known test cases:
- Small semiprimes (< 1000)
- Medium numbers (10-20 digits)
- Large numbers (20-30 digits)
- Edge cases (primes, perfect squares)

### 4. Performance Benchmarks
Measures performance across different number sizes:
- 10-digit numbers
- 15-digit numbers
- 20-digit numbers
- 26-digit challenge number

### 5. Memory Tests
Analyzes memory usage and GPU utilization:
- Per-thread memory requirements
- Maximum allocation capacity
- GPU occupancy calculations
- Thread configuration optimization

### 6. Stress Tests
Tests reliability under extreme conditions:
- Rapid kernel launches
- Maximum thread utilization
- Memory allocation/deallocation cycles
- Long-running factorizations

## Test Cases

### Known Factorizations
| Number | Factors | Description |
|--------|---------|-------------|
| 143 | 11 × 13 | Small semiprime |
| 15482526220500967432610341 | 1804166129797 × 8581541336353 | 26-digit challenge |
| 999999999999999989 | Prime | 18-digit prime test |

### Special Test: 26-Digit Number
The number `15482526220500967432610341` is used as a benchmark:
- Factors: 1804166129797 × 8581541336353
- Both factors are 13-digit primes
- Tests the limits of Pollard's Rho algorithm
- Expected time: ~4 minutes on GTX 2070

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0 or newer
- NVIDIA GPU with Compute Capability 7.0+
- C++14 compatible compiler
- Linux or WSL2 environment

### Build Options
```bash
# Default build (optimized for GTX 2070)
make -f Makefile.v22test

# Clean build artifacts
make -f Makefile.v22test clean

# Build and run specific test
make -f Makefile.v22test test-26digit
```

### Using the Test Runner Script
The `run_v22_tests.sh` script automates the entire test process:

1. Builds the test suite
2. Runs all test categories
3. Captures output and timing information
4. Generates a comprehensive report
5. Creates performance comparison data

Results are saved in a timestamped directory:
```
test_results_v22_YYYYMMDD_HHMMSS/
├── test_log.txt           # Overall test log
├── test_summary.md        # Markdown summary report
├── unit_tests.txt         # Unit test output
├── component_tests.txt    # Component test output
├── integration_tests.txt  # Integration test output
├── benchmarks.txt         # Benchmark results
├── memory_tests.txt       # Memory analysis
├── stress_tests.txt       # Stress test results
├── 26digit_test.txt       # 26-digit challenge result
├── results.csv            # CSV data for analysis
└── compare_v21_v22.py     # Performance comparison script
```

## Interpreting Results

### Success Criteria
- All unit tests must pass
- Component tests verify correct algorithm behavior
- Integration tests validate known factorizations
- Benchmarks should match expected performance (±10%)
- Memory usage should stay within GPU limits
- Stress tests should complete without errors

### Performance Expectations
Based on GTX 2070 (8GB, 2304 cores):
- 10-digit: < 0.2 seconds
- 15-digit: < 3 seconds
- 20-digit: < 20 seconds
- 26-digit: < 300 seconds

### Common Issues
1. **Build failures**: Check CUDA installation and compute capability
2. **Out of memory**: Reduce thread count in test configuration
3. **Timeout**: Some large numbers may exceed test timeouts
4. **Incorrect factors**: Verify uint128_t arithmetic implementation

## Validation

To validate the 26-digit factorization separately:
```bash
# Build validation program
nvcc -O3 -o validate_26digit validate_26digit.cu

# Run validation
./validate_26digit
```

This will verify that:
- 15482526220500967432610341 = 1804166129797 × 8581541336353
- The multiplication is performed correctly
- The factors are valid

## Future Improvements

### v2.3.0 Tests (Planned)
- Quadratic Sieve component tests
- ECM algorithm validation
- Larger number benchmarks (30-40 digits)
- Multi-GPU scaling tests

### v3.0.0 Tests (Planned)
- GNFS implementation validation
- 50+ digit factorization tests
- Distributed computing tests
- Performance comparison with CADO-NFS

## Contributing

When adding new tests:
1. Add test case to appropriate category
2. Update expected results in benchmark_results_v22.md
3. Include timing expectations
4. Document any special requirements
5. Ensure backwards compatibility

## Support

For issues or questions:
1. Check test output logs for detailed error messages
2. Verify GPU compute capability matches build flags
3. Ensure sufficient GPU memory for large tests
4. Review benchmark_results_v22.md for expected behavior