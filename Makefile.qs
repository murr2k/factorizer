# Makefile for Quadratic Sieve implementation
# Builds both standalone and integrated versions

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14 -lineinfo
NVCC_DEBUG_FLAGS = -g -G -DDEBUG

# Source files
QS_CORE_SRC = quadratic_sieve_core.cu
QS_OPT_SRC = quadratic_sieve_optimized.cu
TEST_SRC = test_quadratic_sieve.cu

# Include paths
INCLUDES = -I.

# Targets
all: test_qs qs_standalone qs_optimized

# Test program
test_qs: $(TEST_SRC) $(QS_CORE_SRC) quadratic_sieve.cuh uint128_improved.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(TEST_SRC) $(QS_CORE_SRC)

# Standalone QS implementation
qs_standalone: $(QS_CORE_SRC) quadratic_sieve.cuh uint128_improved.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(QS_CORE_SRC)

# Optimized QS implementation with class interface
qs_optimized: $(QS_OPT_SRC) $(TEST_SRC) quadratic_sieve.cuh uint128_improved.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(TEST_SRC) $(QS_OPT_SRC) $(QS_CORE_SRC)

# Debug versions
debug: NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
debug: test_qs_debug qs_standalone_debug

test_qs_debug: $(TEST_SRC) $(QS_CORE_SRC)
	$(NVCC) $(NVCC_DEBUG_FLAGS) $(INCLUDES) -o $@ $(TEST_SRC) $(QS_CORE_SRC)

qs_standalone_debug: $(QS_CORE_SRC)
	$(NVCC) $(NVCC_DEBUG_FLAGS) $(INCLUDES) -o $@ $(QS_CORE_SRC)

# Performance testing
perf_test: test_qs
	./test_qs perf

# GPU kernel testing
gpu_test: test_qs
	./test_qs gpu

# Integration with main factorizer
integrate: quadratic_sieve.cuh $(QS_OPT_SRC)
	@echo "Quadratic Sieve ready for integration"
	@echo "Include quadratic_sieve.cuh in your main factorizer"
	@echo "Link with $(QS_OPT_SRC) when building"

# Clean
clean:
	rm -f test_qs qs_standalone qs_optimized *_debug *.o

# Help
help:
	@echo "Quadratic Sieve Makefile targets:"
	@echo "  make all          - Build all targets"
	@echo "  make test_qs      - Build test program"
	@echo "  make qs_standalone - Build standalone QS"
	@echo "  make qs_optimized  - Build optimized QS with class interface"
	@echo "  make debug        - Build debug versions"
	@echo "  make perf_test    - Run performance tests"
	@echo "  make gpu_test     - Run GPU kernel tests"
	@echo "  make integrate    - Prepare for integration"
	@echo "  make clean        - Clean build artifacts"

.PHONY: all debug perf_test gpu_test integrate clean help