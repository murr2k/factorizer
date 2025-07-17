#!/bin/bash

echo "=== CUDA Debug Info ==="
echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader

echo -e "\n2. CUDA Runtime Version:"
nvcc --version | grep release

echo -e "\n3. Test CUDA detection directly:"
./test_cuda_detect

echo -e "\n4. Check library loading:"
ldd ./pleiotropy_analyzer | grep -E "(cuda|libcu)"

echo -e "\n5. Run with CUDA_VISIBLE_DEVICES:"
CUDA_VISIBLE_DEVICES=0 ./pleiotropy_analyzer --snps 100 --samples 100 2>&1 | head -5

echo -e "\n6. Test factorizer (which works):"
./factorizer 12345 2>&1 | head -5