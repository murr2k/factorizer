#!/bin/bash
# Wrapper script for running improved factorizer

# Set library paths for WSL2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Run with arguments
exec ./factorizer_improved "$@"
