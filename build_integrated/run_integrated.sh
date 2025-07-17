#!/bin/bash
# Run script for CUDA Factorizer v2.2.0 - Integrated Edition

if [ -z "$1" ]; then
    echo "Usage: $0 <number|test_case> [options]"
    echo ""
    echo "Test cases:"
    echo "  test_26digit    - Test 26-digit case (ECM optimal)"
    echo "  test_86bit      - Test 86-bit case (QS optimal)"
    echo ""
    echo "Options:"
    echo "  -q      Quiet mode"
    echo "  -h      Help"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/factorizer_integrated" "$@"
