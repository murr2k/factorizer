#!/bin/bash

# Full CUDA Demonstration for Genomic Pleiotropy Analysis
# Showcases all features on GTX 2070

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "============================================"
echo "Full CUDA Genomic Pleiotropy Demonstration"
echo "NVIDIA GeForce RTX 2070 - 2304 CUDA Cores"
echo "============================================"

# 1. Performance scaling test
echo -e "\n[1. Performance Scaling Analysis]"
echo "Dataset Size | Time (ms) | GFLOPS | Memory (MB)"
echo "-------------|-----------|---------|------------"

for size in 1000 2000 5000 10000; do
    output=$(LD_LIBRARY_PATH=/usr/lib/wsl/lib ./pleiotropy_analyzer --snps $size --samples $((size/10)) --traits $((size/100)) --rank 10 2>&1 | grep -E "(computation time:|Achieved performance:|Found)")
    time=$(echo "$output" | grep "computation time" | grep -oE "[0-9]+" | head -1)
    gflops=$(echo "$output" | grep "GFLOPS" | grep -oE "[0-9.]+" | head -1)
    memory=$((size * size / 10 * 4 / 1024 / 1024))
    printf "%-12s | %-9s | %-7s | %-10s\n" "${size}x$((size/10))" "$time" "$gflops" "$memory"
done

# 2. Pleiotropic gene detection accuracy
echo -e "\n[2. Pleiotropic Gene Detection]"
echo "Testing with synthetic data containing known pleiotropic patterns..."
./test_pleiotropy

# 3. Memory bandwidth test
echo -e "\n[3. Memory Bandwidth Utilization]"
nvprof --metrics gld_efficiency,gst_efficiency,dram_utilization --log-file bandwidth.log ./pleiotropy_analyzer --snps 5000 --samples 500 --traits 50 --rank 20 2>&1 >/dev/null
if [ -f bandwidth.log ]; then
    echo "Global Load Efficiency: $(grep "gld_efficiency" bandwidth.log | awk '{print $NF}')"
    echo "Global Store Efficiency: $(grep "gst_efficiency" bandwidth.log | awk '{print $NF}')"
    echo "DRAM Utilization: $(grep "dram_utilization" bandwidth.log | awk '{print $NF}')"
    rm -f bandwidth.log
fi

# 4. Large-scale stress test
echo -e "\n[4. Large-Scale Stress Test]"
echo "Processing 20,000 SNPs x 2,000 samples x 100 traits..."
time LD_LIBRARY_PATH=/usr/lib/wsl/lib ./pleiotropy_analyzer --snps 20000 --samples 2000 --traits 100 --rank 30

# 5. Comparison with theoretical peak
echo -e "\n[5. Performance Analysis]"
echo "GTX 2070 Theoretical Peak: 7.5 TFLOPS (FP32)"
echo "Achieved Performance: See results above"
echo "Memory Bandwidth: 448 GB/s theoretical"

echo -e "\n✅ Full demonstration completed successfully!"