# Intelligent Algorithm Selection System

## Overview

The Algorithm Selector is an intelligent system that automatically chooses the optimal factorization method based on the characteristics of the input number. It analyzes various properties of the number and selects the most appropriate algorithm to maximize the chances of finding factors quickly.

## Features

### 1. Number Analysis
- **Bit length and size estimation**: Determines computational complexity
- **Perfect power detection**: Identifies numbers of the form a^b
- **Primality testing**: Uses Miller-Rabin test for probabilistic primality checking
- **Small factor detection**: Quick GPU-accelerated trial division
- **Smoothness estimation**: Evaluates the presence of small prime factors
- **GPU memory assessment**: Adapts to available GPU resources

### 2. Available Algorithms

#### Trial Division
- Best for: Numbers with small prime factors
- Complexity: O(√n)
- GPU-optimized for parallel prime checking

#### Pollard's Rho
- Best for: Numbers with medium-sized factors
- Complexity: O(n^(1/4))
- Implements Brent's improvements and adaptive parameters

#### Quadratic Sieve (Future)
- Best for: Large semiprimes
- Complexity: O(exp(√(ln n ln ln n)))

#### Elliptic Curve Method (Future)
- Best for: Numbers with medium factors (20-40 digits)
- Complexity: O(exp(√(2 ln p ln ln p)))

### 3. Selection Heuristics

The system uses a confidence-based scoring system:

```
Confidence Score Interpretation:
90-100%: Highly confident, factors should be found quickly
70-89%: Good confidence, standard approach should work
50-69%: Moderate confidence, may require extended computation
20-49%: Low confidence, number may be prime or hard semiprime
0-19%: Very low confidence, specialized algorithms needed
```

### 4. Dynamic Algorithm Switching

The system monitors progress and can switch algorithms if:
- No progress for extended periods
- Time estimates are significantly exceeded
- GPU memory pressure is detected
- Better algorithm becomes apparent during factorization

## Usage

### Building

```bash
make -f Makefile.intelligent
```

### Running the Algorithm Selector

```bash
# Analyze a number and get algorithm recommendation
./algorithm_selector <number>

# Example
./algorithm_selector 15482526220500967432610341
```

### Using the Intelligent Factorizer

```bash
# Factor a number with automatic algorithm selection
./factorizer_intelligent <number> [--quiet]

# Examples
./factorizer_intelligent 1234567890123456789
./factorizer_intelligent 15482526220500967432610341 --quiet
```

## API Integration

Include the header file and use the analysis functions:

```cpp
#include "algorithm_selector.cuh"

// Analyze a number
uint128_t n = parse_decimal("1234567890");
NumberAnalysis analysis = analyze_number(n);

// Get algorithm recommendation
AlgorithmChoice choice = select_algorithm(analysis);

// Monitor progress
ProgressEstimate progress = estimate_progress(choice, analysis, 
                                            elapsed_ms, iterations);
```

## Algorithm Selection Examples

### Example 1: Perfect Power
```
Input: 1024
Analysis: Perfect power detected (2^10)
Strategy: Factor the base
Algorithm: Trial Division
Confidence: 95%
```

### Example 2: Small Smooth Number
```
Input: 30030
Analysis: High smoothness (many small factors)
Strategy: Extensive trial division
Algorithm: Trial Division
Confidence: 85%
```

### Example 3: Large Semiprime
```
Input: 15482526220500967432610341
Analysis: 84-bit number, no small factors
Strategy: Combined approach
Algorithm: Trial Division + Pollard's Rho
Confidence: 60%
```

### Example 4: Probable Prime
```
Input: 1000000007
Analysis: Likely prime (Miller-Rabin)
Strategy: Extended testing
Algorithm: Pollard's Rho with high iterations
Confidence: 20%
```

## Performance Characteristics

### Time Complexity by Number Type

| Number Type | Algorithm | Expected Time |
|------------|-----------|---------------|
| Small (<32 bits) | Trial Division | <100ms |
| Perfect Power | Power Detection | <10ms |
| Smooth Number | Trial Division | <500ms |
| Medium Composite | Pollard's Rho | 1-5s |
| Large Semiprime | Combined | 10-60s |
| Prime | All methods | Timeout |

### GPU Memory Usage

- Small numbers: <100 MB
- Medium numbers: 100-500 MB
- Large numbers: 500 MB - 2 GB
- Adaptive based on available GPU memory

## Future Enhancements

1. **Quadratic Sieve Implementation**
   - For numbers > 100 bits
   - Sieving with GPU acceleration

2. **Elliptic Curve Method**
   - For finding medium-sized factors
   - Multiple curves in parallel

3. **Number Field Sieve**
   - For very large numbers (>110 digits)
   - Distributed computation support

4. **Machine Learning Integration**
   - Train on factorization patterns
   - Improve time estimates

5. **Batch Processing**
   - Factor multiple numbers simultaneously
   - Optimize GPU utilization

## Troubleshooting

### "No factor found"
- Number might be prime
- Increase iteration limits
- Try different random seeds

### "Low confidence score"
- Number has special structure
- May need specialized algorithms
- Consider manual algorithm override

### "GPU memory error"
- Reduce thread/block count
- Use CPU fallback for very large numbers
- Check available GPU memory

## References

1. Pollard, J.M. (1975). "A Monte Carlo method for factorization"
2. Brent, R.P. (1980). "An improved Monte Carlo factorization algorithm"
3. Cohen, H. (1993). "A Course in Computational Algebraic Number Theory"
4. Pomerance, C. (1985). "The Quadratic Sieve Factoring Algorithm"