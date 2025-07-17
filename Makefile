# Makefile for CUDA Genomic Pleiotropy Analysis
# Optimized for NVIDIA GTX 2070

NVCC = nvcc
CXX = g++
CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# GTX 2070 uses compute capability 7.5 (Turing architecture)
CUDA_ARCH = -arch=sm_75

# Compiler flags
NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++14 --use_fast_math -Xcompiler -fopenmp
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
NVCC_FLAGS += -I$(CUDA_INCLUDE)

# Linker flags
LDFLAGS = -L$(CUDA_LIB) -lcudart -lcublas -lcurand -lcusolver -lgomp

# Source files
CUDA_SOURCES = pleiotropy_cuda.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Target executable
TARGET = pleiotropy_analyzer

# Build rules
all: $(TARGET)

$(TARGET): $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Additional tools
benchmark: $(TARGET)
	./$(TARGET) --benchmark

profile: $(TARGET)
	nvprof --print-gpu-trace ./$(TARGET)

memcheck: $(TARGET)
	cuda-memcheck ./$(TARGET)

clean:
	rm -f $(CUDA_OBJECTS) $(TARGET)

# Advanced factorization module (if you need the large number factorization)
factorizer: factorizer_cuda.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS) -lgmp

factorizer_cuda.o: factorizer_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 128-bit factorization module
factorizer128: factorizer_cuda_128.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS) -lgmp

factorizer_cuda_128.o: factorizer_cuda_128.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# QA test suite
test128: test_128bit.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS) -lgmp

test_128bit.o: test_128bit.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# v2.1 128-bit factorizer
factorizer_v21_128bit: factorizer_v21_128bit.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

factorizer_v21_128bit.o: factorizer_v21_128bit.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# v2.1 128-bit factorizer diagnostic version
factorizer_v21_128bit_diagnostic: factorizer_v21_128bit_diagnostic.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

factorizer_v21_128bit_diagnostic.o: factorizer_v21_128bit_diagnostic.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 128-bit arithmetic test suite
test_128bit_arithmetic: test_128bit_arithmetic.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

test_128bit_arithmetic.o: test_128bit_arithmetic.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Pollard's rho algorithm test
test_pollard_rho: test_pollard_rho.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

test_pollard_rho.o: test_pollard_rho.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

.PHONY: all clean benchmark profile memcheck test-all test-arithmetic test-pollard

test-all: test128 factorizer128
	./test128
	./factorizer128 94498503396937386863845286721509

test-arithmetic: test_128bit_arithmetic
	./test_128bit_arithmetic

test-pollard: test_pollard_rho
	./test_pollard_rho

# Barrett reduction test
test_barrett_clean: test_barrett_clean.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

test_barrett_clean.o: test_barrett_clean.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

test-barrett: test_barrett_clean
	./test_barrett_clean