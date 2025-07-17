#!/bin/bash

# Comprehensive benchmark script for genomic pleiotropy CUDA implementation
# Tests performance across various data sizes and configurations

echo "==================================="
echo "Genomic Pleiotropy CUDA Benchmark"
echo "GPU: NVIDIA GTX 2070"
echo "==================================="

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# Display GPU information
echo -e "\n[GPU Information]"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

# Compile the project
echo -e "\n[Compilation]"
make clean
make all

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Create results directory
mkdir -p benchmark_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="benchmark_results/benchmark_${TIMESTAMP}.csv"

# CSV header
echo "Test,SNPs,Samples,Traits,Rank,Time(ms),Memory(MB),GFLOPS,Precision,Recall" > $RESULT_FILE

# Run benchmarks with different configurations
echo -e "\n[Running Benchmarks]"

# Small-scale tests
echo -e "\n--- Small Scale (Development) ---"
./run_pleiotropy.sh --benchmark --snps 1000 --samples 100 --traits 10 --rank 5 >> $RESULT_FILE
./run_pleiotropy.sh --benchmark --snps 2000 --samples 200 --traits 20 --rank 10 >> $RESULT_FILE

# Medium-scale tests
echo -e "\n--- Medium Scale (Research) ---"
./run_pleiotropy.sh --benchmark --snps 5000 --samples 500 --traits 50 --rank 15 >> $RESULT_FILE
./run_pleiotropy.sh --benchmark --snps 10000 --samples 1000 --traits 100 --rank 20 >> $RESULT_FILE

# Large-scale tests
echo -e "\n--- Large Scale (Production) ---"
./run_pleiotropy.sh --benchmark --snps 20000 --samples 2000 --traits 200 --rank 30 >> $RESULT_FILE
./run_pleiotropy.sh --benchmark --snps 50000 --samples 5000 --traits 500 --rank 50 >> $RESULT_FILE

# Extreme-scale test (if memory permits)
echo -e "\n--- Extreme Scale (Stress Test) ---"
./run_pleiotropy.sh --benchmark --snps 100000 --samples 10000 --traits 1000 --rank 100 >> $RESULT_FILE 2>&1

# Memory optimization tests
echo -e "\n[Memory Access Pattern Tests]"
echo -e "\n--- Coalesced vs Non-coalesced Access ---"
nvprof --metrics gld_efficiency,gst_efficiency ./run_pleiotropy.sh --test-memory

echo -e "\n--- Cache Hit Rates ---"
nvprof --metrics l1_cache_global_hit_rate,l2_l1_read_hit_rate ./run_pleiotropy.sh --test-cache

# Factorization performance
echo -e "\n[Factorization Benchmarks]"
if [ -f "./factorizer" ]; then
    echo -e "\n--- Testing with genomic sequences ---"
    # 40-digit semiprime example
    ./factorizer "1234567890123456789012345678901234567891"
    
    # Test with actual genomic sequence mapping
    SEQUENCE="ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    echo "Testing genomic sequence: $SEQUENCE"
    ./factorizer --sequence "$SEQUENCE"
fi

# Generate performance plots (requires Python with matplotlib)
if command -v python3 &> /dev/null; then
    echo -e "\n[Generating Performance Plots]"
    python3 << EOF
import csv
import matplotlib.pyplot as plt
import numpy as np

# Read benchmark results
data = []
with open('$RESULT_FILE', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            data.append({
                'snps': int(row['SNPs']),
                'time': float(row['Time(ms)']),
                'memory': float(row['Memory(MB)']),
                'gflops': float(row['GFLOPS'])
            })
        except:
            pass

if data:
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Time vs Problem Size
    snps = [d['snps'] for d in data]
    times = [d['time'] for d in data]
    axes[0,0].plot(snps, times, 'b-o')
    axes[0,0].set_xlabel('Number of SNPs')
    axes[0,0].set_ylabel('Time (ms)')
    axes[0,0].set_title('Computation Time Scaling')
    axes[0,0].grid(True)
    
    # Plot 2: Memory Usage
    memory = [d['memory'] for d in data]
    axes[0,1].plot(snps, memory, 'r-o')
    axes[0,1].set_xlabel('Number of SNPs')
    axes[0,1].set_ylabel('Memory (MB)')
    axes[0,1].set_title('Memory Usage')
    axes[0,1].grid(True)
    
    # Plot 3: GFLOPS Performance
    gflops = [d['gflops'] for d in data]
    axes[1,0].plot(snps, gflops, 'g-o')
    axes[1,0].set_xlabel('Number of SNPs')
    axes[1,0].set_ylabel('GFLOPS')
    axes[1,0].set_title('Computational Performance')
    axes[1,0].grid(True)
    
    # Plot 4: Efficiency
    theoretical_gflops = 7.5 * 1000  # GTX 2070 theoretical peak
    efficiency = [g/theoretical_gflops * 100 for g in gflops]
    axes[1,1].plot(snps, efficiency, 'm-o')
    axes[1,1].set_xlabel('Number of SNPs')
    axes[1,1].set_ylabel('Efficiency (%)')
    axes[1,1].set_title('GPU Utilization Efficiency')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/performance_${TIMESTAMP}.png')
    print("Performance plots saved to benchmark_results/performance_${TIMESTAMP}.png")
EOF
fi

# Summary report
echo -e "\n[Benchmark Summary]"
echo "Results saved to: $RESULT_FILE"
echo -e "\nTop Performance Metrics:"
tail -n +2 $RESULT_FILE | sort -t',' -k6 -n | head -5

echo -e "\n✅ Benchmark completed successfully!"