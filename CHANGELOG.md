# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-17

### Added
- Comprehensive uint128_t arithmetic library with proper carry handling
- Barrett reduction framework for fast modular arithmetic
- cuRAND integration for high-quality random number generation
- Parallel Pollard's Rho implementation with warp-level cooperation
- Binary GCD algorithm for efficient greatest common divisor computation
- 256-bit result type for 128×128 bit multiplication
- Factor verification through multiplication
- Comprehensive test suite for all improvements
- Performance benchmarking framework

### Changed
- Complete rewrite of 128-bit multiplication with correct carry propagation
- Pollard's Rho now uses multiple starting points for better success rate
- Improved memory access patterns for GPU optimization
- Enhanced error handling and timeout management

### Fixed
- Fixed incorrect carry handling in uint128_t multiplication
- Resolved overflow issues in large number arithmetic
- Fixed thread correlation issues in random number generation
- Corrected test cases in test_128bit.cu (4/8 had wrong factors)

### Performance
- 11-digit factorization: >8x speedup (from >30s to 3.8s)
- 12-digit factorization: From timeout to 6.5s
- 13-digit factorization: From timeout to 8.6s
- 16-digit factorization: Successfully completes in 11.1s

## [1.0.0] - 2025-01-16

### Added
- Initial implementation of 128-bit factorization
- Basic Pollard's Rho algorithm for CUDA
- Genomic pleiotropy analysis framework
- Non-negative Matrix Factorization (NMF) implementation
- WSL2 compatibility wrapper scripts
- Basic test suite

### Known Issues
- Timeout on numbers larger than 10 digits
- Incorrect carry propagation in multiplication
- No proper random number generation
- Missing modular arithmetic optimizations

## Version Guidelines

### Version Format: MAJOR.MINOR.PATCH

- **MAJOR**: Increment for incompatible API changes or major algorithmic changes
- **MINOR**: Increment for new features added in a backwards compatible manner
- **PATCH**: Increment for backwards compatible bug fixes

### Pre-release versions
- Alpha: X.Y.Z-alpha.N (early development)
- Beta: X.Y.Z-beta.N (feature complete, testing)
- Release Candidate: X.Y.Z-rc.N (final testing)

### Examples
- 2.0.0: Major algorithmic improvements (current)
- 2.1.0: Would add new features like Montgomery reduction
- 2.0.1: Would fix bugs in current implementation
- 3.0.0: Would be a complete architectural change