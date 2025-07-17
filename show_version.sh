#!/bin/bash

# Display version information for the Factorizer project

echo "==================================="
echo "   CUDA 128-bit Factorizer"
echo "==================================="
echo

# Read version from VERSION file
if [ -f VERSION ]; then
    VERSION=$(cat VERSION)
    echo "Version: $VERSION"
else
    echo "Version: Unknown (VERSION file not found)"
fi

echo "Release Date: 2025-01-17"
echo "Codename: Hive-Mind"
echo

echo "Key Features in this release:"
echo "  ✓ Improved uint128_t arithmetic with proper carry handling"
echo "  ✓ Barrett reduction framework for fast modular operations"
echo "  ✓ cuRAND integration for better randomness"
echo "  ✓ Parallel Pollard's Rho with warp-level cooperation"
echo "  ✓ 8x+ performance improvement on 11-digit numbers"
echo

echo "Tested on: NVIDIA GeForce RTX 2070"
echo "CUDA Version: 11.0+"
echo

echo "For more details, see:"
echo "  - README.md (Release Notes section)"
echo "  - CHANGELOG.md (Detailed version history)"
echo "  - FINAL_TEST_RESULTS.md (Performance benchmarks)"
echo
echo "==================================="