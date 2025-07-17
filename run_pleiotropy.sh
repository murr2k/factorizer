#!/bin/bash

# CUDA environment wrapper for WSL2
# Ensures proper CUDA runtime initialization

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Force CUDA device visibility
export CUDA_VISIBLE_DEVICES=0

# WSL2 specific: ensure NVIDIA runtime is accessible
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Run the analyzer with all arguments passed through
exec ./pleiotropy_analyzer "$@"