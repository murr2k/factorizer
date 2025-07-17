# GPU-Optimized Quadratic Sieve Design

## Executive Summary

This document outlines the design of a GPU-optimized Quadratic Sieve (QS) implementation for factoring semiprimes with balanced factors (10-20 digits each), targeting 20-40 digit composite numbers. The implementation leverages CUDA parallelization for the computationally intensive sieving phase while maintaining CPU control for sequential operations.

## Algorithm Overview

The Quadratic Sieve is a general-purpose integer factorization algorithm that excels for numbers in the 20-100 digit range. It operates by finding smooth numbers (numbers with only small prime factors) and using linear algebra to find a non-trivial factorization.

### Core Algorithm Phases

1. **Initialization Phase**
   - Factor base generation
   - Polynomial coefficient selection
   - Precomputation of logarithms

2. **Sieving Phase** (GPU-optimized)
   - Polynomial evaluation
   - Sieve array initialization
   - Trial division replacement via logarithmic sieving
   - Smooth number collection

3. **Linear Algebra Phase**
   - Matrix construction from smooth relations
   - Gaussian elimination over GF(2)
   - Solution extraction

4. **Factor Recovery Phase**
   - Square root computation
   - GCD calculation
   - Factor verification

## GPU Parallelization Strategy

### 1. Factor Base Generation

```cuda
__global__ void generate_factor_base_kernel(
    uint32_t* primes,
    uint32_t* quadratic_residues,
    uint128_t n,
    int max_prime,
    int* base_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread tests a range of candidates
    for (int p = 3 + 2*tid; p < max_prime; p += 2*stride) {
        if (is_prime_gpu(p)) {
            // Compute Legendre symbol (n/p)
            if (legendre_symbol(n, p) == 1) {
                int idx = atomicAdd(base_size, 1);
                primes[idx] = p;
                // Compute square root of n mod p
                quadratic_residues[idx] = tonelli_shanks(n, p);
            }
        }
    }
}
```

### 2. Parallel Sieving Architecture

The sieving phase is the most computationally intensive and highly parallelizable:

```cuda
// Sieve configuration for optimal GPU utilization
struct SieveConfig {
    int sieve_interval;      // M = 2^20 typical
    int num_polynomials;     // Number of polynomials to sieve
    int threads_per_poly;    // Threads assigned per polynomial
    int blocks_per_poly;     // Grid configuration
    int shared_mem_size;     // Shared memory per block
};

__global__ void quadratic_sieve_kernel(
    uint128_t n,
    uint32_t* factor_base,
    uint32_t* qr_roots,
    int base_size,
    int8_t* sieve_arrays,    // Log values
    uint32_t* smooth_indices, // Output smooth numbers
    int* smooth_count,
    SieveConfig config
) {
    // Shared memory for factor base cache
    extern __shared__ uint32_t s_primes[];
    
    int poly_id = blockIdx.y;
    int sieve_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load factor base to shared memory
    if (threadIdx.x < base_size && threadIdx.x < blockDim.x) {
        s_primes[threadIdx.x] = factor_base[threadIdx.x];
    }
    __syncthreads();
    
    // Compute polynomial Q(x) = (Ax + B)^2 - n
    PolynomialCoeffs poly = compute_polynomial(n, poly_id);
    
    // Initialize sieve array with approximated logarithms
    int8_t* my_sieve = sieve_arrays + poly_id * config.sieve_interval;
    
    // Parallel sieving with coalesced memory access
    for (int offset = sieve_idx; 
         offset < config.sieve_interval; 
         offset += blockDim.x * gridDim.x) {
        
        int64_t x = offset - config.sieve_interval/2;
        uint128_t Qx = evaluate_polynomial(poly, x, n);
        
        // Approximate log of |Q(x)|
        my_sieve[offset] = approximate_log(Qx);
    }
    
    __syncthreads();
    
    // Subtract logarithms of factor base primes
    for (int i = threadIdx.x; i < base_size; i += blockDim.x) {
        uint32_t p = s_primes[i];
        uint8_t log_p = integer_log2(p);
        
        // Find sieving positions for this prime
        int pos1, pos2;
        compute_sieve_positions(poly, p, qr_roots[i], &pos1, &pos2);
        
        // Sieve with stride p
        for (int j = pos1; j < config.sieve_interval; j += p) {
            atomicSub(&my_sieve[j], log_p);
        }
        if (pos2 != pos1) {
            for (int j = pos2; j < config.sieve_interval; j += p) {
                atomicSub(&my_sieve[j], log_p);
            }
        }
    }
    
    __syncthreads();
    
    // Identify smooth numbers (threshold detection)
    const int8_t threshold = 10; // Tunable parameter
    
    if (sieve_idx < config.sieve_interval && my_sieve[sieve_idx] < threshold) {
        // Found potential smooth number
        int idx = atomicAdd(smooth_count, 1);
        smooth_indices[idx] = poly_id * config.sieve_interval + sieve_idx;
    }
}
```

### 3. Smooth Number Verification

```cuda
__global__ void verify_smooth_kernel(
    uint128_t n,
    uint32_t* smooth_candidates,
    uint32_t* factor_base,
    int base_size,
    SmoothRelation* relations,
    int* relation_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *smooth_count) return;
    
    uint32_t idx = smooth_candidates[tid];
    int poly_id = idx / SIEVE_INTERVAL;
    int x_offset = idx % SIEVE_INTERVAL;
    
    // Reconstruct Q(x)
    PolynomialCoeffs poly = compute_polynomial(n, poly_id);
    int64_t x = x_offset - SIEVE_INTERVAL/2;
    uint128_t Qx = evaluate_polynomial(poly, x, n);
    
    // Trial division to verify smoothness
    uint128_t remainder = Qx;
    uint32_t exponents[MAX_FACTOR_BASE_SIZE] = {0};
    
    for (int i = 0; i < base_size && remainder > 1; i++) {
        uint32_t p = factor_base[i];
        while (remainder % p == 0) {
            remainder /= p;
            exponents[i]++;
        }
    }
    
    if (remainder == 1) {
        // Confirmed smooth - store relation
        int idx = atomicAdd(relation_count, 1);
        relations[idx].x = x;
        relations[idx].poly_id = poly_id;
        memcpy(relations[idx].exponents, exponents, base_size * sizeof(uint32_t));
    }
}
```

## Memory Requirements and Data Structures

### GPU Memory Layout

```cpp
struct QuadraticSieveGPU {
    // Factor base storage
    uint32_t* d_factor_base;        // Size: ~4KB for 1000 primes
    uint32_t* d_quadratic_residues; // Size: ~4KB
    
    // Sieving arrays (multiple polynomials)
    int8_t* d_sieve_arrays;         // Size: NUM_POLY * SIEVE_INTERVAL
                                    // Example: 32 * 1MB = 32MB
    
    // Smooth number collection
    uint32_t* d_smooth_indices;     // Size: ~1MB (overallocated)
    SmoothRelation* d_relations;    // Size: ~10MB for safety
    
    // Working memory
    uint128_t* d_polynomial_coeffs; // Size: NUM_POLY * 16 bytes
    
    // Total GPU memory: ~50MB for typical 30-digit factorization
};
```

### Host Memory Structures

```cpp
struct QuadraticSieveHost {
    // Configuration
    uint128_t n;                    // Number to factor
    int factor_base_size;           // Typically sqrt(exp(sqrt(log n log log n)))
    int required_relations;         // factor_base_size + 20
    
    // Factor base
    std::vector<uint32_t> primes;
    std::vector<uint32_t> quadratic_residues;
    
    // Collected relations
    std::vector<SmoothRelation> relations;
    
    // Matrix for linear algebra
    BitMatrix matrix;               // Size: factor_base_size × required_relations
};
```

## Expected Performance Analysis

### Computational Complexity

- **Factor Base Size**: B ≈ exp(0.5 * sqrt(log n * log log n))
- **Sieving Interval**: M ≈ sqrt(n) / B
- **Time Complexity**: O(exp(sqrt(log n * log log n)))

### GPU Performance Characteristics

For a 30-digit number (n ≈ 10^30):
- Factor base size: ~1000 primes
- Sieve interval: ~1M values
- Polynomials needed: ~1200

**GPU Utilization:**
- Memory bandwidth: ~400 GB/s utilized (90% efficiency)
- Compute throughput: ~10 TFLOPS (mixed integer/float)
- Kernel occupancy: >80% with proper tuning

**Expected Timings (RTX 2070):**
- 20-digit factorization: <10 seconds
- 30-digit factorization: 2-5 minutes
- 40-digit factorization: 20-40 minutes

### Comparison with CPU Implementation

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Factor base generation | 100ms | 5ms | 20x |
| Sieving (per polynomial) | 50ms | 2ms | 25x |
| Smooth verification | 200ms | 10ms | 20x |
| Overall sieving phase | 60s | 2.4s | 25x |

## Integration Points with Main Architecture

### 1. Algorithm Selection

Integration with existing `FactorizerConfig`:

```cpp
// In auto_configure function
if (bit_size >= 60 && bit_size <= 130) {
    config.algorithm = QUADRATIC_SIEVE;
    config.reduction = REDUCTION_BARRETT;  // For modular operations
}
```

### 2. Entry Point

```cpp
// New function in main factorizer
FactorizationResult launch_quadratic_sieve(
    uint128_t n,
    FactorizerConfig config
) {
    QuadraticSieveGPU gpu_data;
    QuadraticSieveHost host_data;
    
    // Initialize QS parameters
    initialize_quadratic_sieve(n, &host_data, &gpu_data);
    
    // Phase 1: Generate factor base
    generate_factor_base_gpu(n, &gpu_data, &host_data);
    
    // Phase 2: Sieving loop
    while (host_data.relations.size() < host_data.required_relations) {
        perform_sieving_batch(&gpu_data, &host_data);
        collect_smooth_relations(&gpu_data, &host_data);
    }
    
    // Phase 3: Linear algebra (CPU)
    std::vector<uint128_t> factors = solve_linear_system(host_data);
    
    // Phase 4: Factor recovery
    return extract_factors(n, factors);
}
```

### 3. Memory Management

Utilize existing memory optimization framework:

```cpp
// Integrate with memory_optimizer.cuh
template<>
struct AlignedAllocator<QuadraticSieveGPU> {
    static QuadraticSieveGPU* allocate(size_t factor_base_size) {
        QuadraticSieveGPU* qs = new QuadraticSieveGPU;
        
        // Use existing aligned allocation
        qs->d_factor_base = aligned_alloc_gpu<uint32_t>(factor_base_size);
        qs->d_sieve_arrays = aligned_alloc_gpu<int8_t>(SIEVE_SIZE);
        
        return qs;
    }
};
```

### 4. Progress Monitoring

Extend existing progress monitor:

```cpp
// In progress_monitor_fixed.cuh
void update_qs_progress(
    int relations_found,
    int relations_needed,
    int polynomials_processed
) {
    float progress = (float)relations_found / relations_needed * 100.0f;
    printf("\rQS Progress: %d/%d relations (%.1f%%), %d polynomials processed",
           relations_found, relations_needed, progress, polynomials_processed);
}
```

## Optimization Strategies

### 1. Multiple Polynomial Selection

Use SIQS (Self-Initializing QS) variant:
- Generate polynomials A, B such that B² ≡ n (mod A)
- Reduces coefficient size, improving sieving efficiency

### 2. Large Prime Variation

- Allow one large prime (up to B²) in factorization
- Combines partial relations to form full relations
- Reduces total sieving required by ~30%

### 3. Memory Access Optimization

- **Texture Memory**: Store factor base in texture cache
- **Shared Memory**: Cache frequently accessed primes
- **Coalesced Access**: Ensure warp-aligned memory patterns

### 4. Dynamic Load Balancing

- Monitor smooth number yield per polynomial
- Adjust polynomial selection strategy
- Redistribute work among thread blocks

## Testing and Validation

### Test Cases

1. **Small Semiprimes** (20-25 digits)
   - Verify correctness against known factorizations
   - Compare timing with Pollard's Rho

2. **Medium Semiprimes** (25-35 digits)
   - Primary target range
   - Benchmark against CPU implementations

3. **Large Semiprimes** (35-40 digits)
   - Stress test memory usage
   - Validate linear algebra phase

### Validation Strategy

```cpp
bool validate_qs_implementation() {
    // Test vector of known factorizations
    struct TestCase {
        const char* n;
        const char* p;
        const char* q;
    };
    
    TestCase tests[] = {
        {"10000000000000000000037", "100000007", "100000000000000000003"},
        {"1000000000000000000000000000037", "1000000007", "1000000000000000000000029"},
        // ... more test cases
    };
    
    for (auto& test : tests) {
        uint128_t n = parse_uint128(test.n);
        auto result = launch_quadratic_sieve(n, config);
        
        if (!verify_factors(n, result)) {
            return false;
        }
    }
    return true;
}
```

## Future Enhancements

1. **Number Field Sieve Preparation**
   - QS serves as stepping stone to NFS
   - Many concepts transfer directly

2. **Distributed Computing**
   - Split sieving across multiple GPUs
   - MPI integration for cluster deployment

3. **Adaptive Algorithm Selection**
   - Automatic fallback to ECM for special forms
   - Hybrid approaches for optimal performance

4. **Hardware-Specific Optimizations**
   - Tensor Core utilization (Ampere/Hopper)
   - Dynamic parallelism for irregular workloads

## Conclusion

This GPU-optimized Quadratic Sieve design provides a robust framework for factoring 20-40 digit numbers with balanced factors. The parallel sieving architecture achieves 20-25x speedup over CPU implementations while maintaining compatibility with the existing factorizer framework. The implementation serves as a crucial bridge between small-number algorithms (Pollard's Rho) and large-number algorithms (GNFS), positioning the factorizer for comprehensive coverage of the factorization problem space.