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
