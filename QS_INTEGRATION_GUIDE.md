# Quadratic Sieve Integration Guide for CUDA Factorizer v2.2.0

## Overview

This guide details the integration of the Quadratic Sieve (QS) algorithm into the CUDA Factorizer v2.2.0 unified framework. The QS implementation is optimized for factoring numbers with large balanced factors (40+ bits each).

## Current State Analysis

### Existing Components

1. **quadratic_sieve_core.cu** - Basic QS implementation with:
   - Factor base generation using Tonelli-Shanks
   - GPU sieving kernel with logarithmic sieving
   - Smooth number detection
   - Trial division verification
   - Simple polynomial (x² - n)

2. **factorizer_v22_fixed.cu** - Unified framework with:
   - Algorithm selection logic
   - Progress monitoring
   - Fallback mechanisms
   - Support for Trial Division and Pollard's Rho variants

### Missing Components

1. **Multiple Polynomial Generation** - MPQS self-initializing polynomials
2. **Matrix Construction** - Building exponent matrix from relations
3. **Linear Algebra** - Gaussian elimination over GF(2)
4. **Square Root Extraction** - Final step to compute factors
5. **QS Integration** - Hooking QS into the unified framework

## Implementation Details

### 1. Complete QS Implementation

Created `quadratic_sieve_complete.cu` with:

```cuda
// Enhanced polynomial generation
bool qs_generate_polynomial(QSContext* ctx, int poly_index) {
    // Implements MPQS-style polynomial selection
    // Form: Q(x) = (ax + b)² - n
    // Where a is product of factor base primes
}

// Optimized GPU sieving
__global__ void qs_sieve_kernel_optimized(QSSieveData data) {
    // Uses shared memory for factor base
    // Handles multiple polynomials
    // Efficient logarithmic sieving
}

// Matrix solving over GF(2)
bool qs_solve_matrix(QSContext* ctx, std::vector<std::vector<int>>& deps) {
    // Gaussian elimination to find null space
    // Identifies linear dependencies
}
```

### 2. Integration Architecture

Created `factorizer_v22_qs_integration.cu` with:

```cuda
class AlgorithmSelectorQS : public AlgorithmSelector {
    // Enhanced selector that considers QS for appropriate numbers
    // Heuristics based on number size and factor balance
};

bool UnifiedFactorizer::run_quadratic_sieve(uint128_t n, const AlgorithmConfig& cfg) {
    // Integrates QS into unified framework
    // Handles progress reporting
    // Manages GPU resources
}
```

### 3. Algorithm Selection Strategy

QS is selected when:
- Number size: 80-120 bits
- Pollard's Rho timeout suggests balanced factors
- No small factors found by trial division

Fallback sequence for 80-120 bit numbers:
1. Pollard's Rho Brent (30 seconds)
2. Quadratic Sieve (5 minutes)
3. Parallel Pollard's Rho (10 minutes)

## GPU Optimization Strategies

### 1. Sieving Optimizations

- **Shared Memory**: Cache factor base subset per block
- **Coalesced Access**: Arrange sieve array for optimal memory patterns
- **Atomic Operations**: Use atomicAdd for logarithm accumulation
- **Block Partitioning**: Distribute primes across blocks efficiently

### 2. Parallelization Approach

```cuda
// Multiple polynomials processed concurrently
Block 0-3: Polynomial 0 sieving
Block 4-7: Polynomial 1 sieving
...

// Within each polynomial
Thread 0-255: Handle primes 0-255
Thread 256-511: Handle primes 256-511
...
```

### 3. Memory Management

- **Pinned Memory**: For factor base and smooth relations
- **Texture Memory**: Consider for read-only factor base access
- **Constant Memory**: For small, frequently accessed parameters

## Building and Testing

### Build Commands

```bash
# Build QS-integrated factorizer
make -f Makefile_qs

# Run integration tests
./factorizer_v22_qs

# Test specific number
./factorizer_v22_qs 29318992932113061061655073

# Run benchmarks
./factorizer_v22_qs benchmark
```

### Test Cases

1. **60-bit with 30-bit factors**: 1152921504606846999
2. **80-bit with 40-bit factors**: 1208925819614629174706449
3. **86-bit test case**: 29318992932113061061655073
4. **26-digit challenge**: 15482526220500967432610341

## Integration Checklist

- [x] Complete polynomial generation (MPQS-style)
- [x] Enhanced GPU sieving kernel
- [x] Matrix construction from relations
- [x] Gaussian elimination solver
- [x] Factor extraction from dependencies
- [x] Integration with AlgorithmSelector
- [x] run_quadratic_sieve() implementation
- [x] Progress monitoring for QS
- [x] Fallback mechanism with QS
- [ ] Performance tuning for specific GPU architectures
- [ ] Large prime variation support
- [ ] Block Lanczos for large matrices

## Performance Expectations

For numbers with balanced factors:
- 60-bit: < 1 second
- 80-bit: 5-30 seconds
- 100-bit: 1-10 minutes
- 120-bit: 10-60 minutes

QS outperforms Pollard's Rho when:
- Both factors > 35 bits
- Factors are roughly equal size
- Number has no small factors

## Future Enhancements

1. **Large Prime Variation**: Handle partial relations with one large prime
2. **Double Large Prime**: Further improvement for sparse relations
3. **Block Lanczos**: More efficient than Gaussian elimination for large matrices
4. **Self-Tuning Parameters**: Automatic adjustment based on GPU and number characteristics
5. **Multi-GPU Support**: Distribute sieving across multiple GPUs

## Debugging Tips

1. **Verify Factor Base**: Check quadratic residues are computed correctly
2. **Monitor Smooth Relations**: Ensure sufficient relations are found
3. **Matrix Rank**: Verify matrix has expected rank
4. **GPU Errors**: Use cuda-memcheck for memory issues
5. **Progress Stalls**: Check if sieving intervals are productive

## Conclusion

The Quadratic Sieve integration adds a powerful algorithm for factoring numbers with large balanced factors. Combined with the existing Pollard's Rho implementations, the factorizer now handles a wider range of factorization challenges efficiently.