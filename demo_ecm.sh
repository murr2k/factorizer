#!/bin/bash
# Demo script for ECM factorization

echo "CUDA Factorizer - Elliptic Curve Method Demo"
echo "============================================"
echo

# First build the programs
echo "Building ECM factorizer..."
./build_ecm.sh
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo
echo "Demo 1: Small semiprime (product of two ~20-bit primes)"
echo "Number: 1099511627791 = 1048583 × 1048573"
echo "Running: ./factorizer_ecm -a ecm 1099511627791"
echo "---"
./factorizer_ecm -a ecm 1099511627791

echo
echo "Demo 2: Larger semiprime (product of two 32-bit primes)"
echo "Number: 18446744073709551557 = 4294967291 × 4294967279"
echo "Running: ./factorizer_ecm -a ecm 18446744073709551557"
echo "---"
./factorizer_ecm -a ecm 18446744073709551557

echo
echo "Demo 3: Auto-selection (let the program choose the best algorithm)"
echo "Number: 123456789012345678901"
echo "Running: ./factorizer_ecm 123456789012345678901"
echo "---"
./factorizer_ecm 123456789012345678901

echo
echo "Demo 4: Running ECM test suite"
echo "This will test ECM on various number sizes and show performance metrics"
echo "Running: ./test_ecm"
echo "---"
./test_ecm

echo
echo "Demo complete!"
echo
echo "Key ECM features demonstrated:"
echo "  • Parallel processing of multiple curves on GPU"
echo "  • Montgomery curve arithmetic for efficiency"
echo "  • Stage 1: Multiplication by smooth numbers"
echo "  • Stage 2: Baby-step giant-step algorithm"
echo "  • Automatic curve generation and testing"
echo
echo "ECM is particularly effective for:"
echo "  • Numbers with medium-sized factors (10-20 digits)"
echo "  • When Pollard's Rho struggles with larger factors"
echo "  • Parallelizable factorization attempts"