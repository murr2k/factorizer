CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17
LIBS = -lgmp -lgmpxx -fopenmp

all: semiprime_factor test_suite

semiprime_factor: semiprime_factor.cpp
	$(CXX) $(CXXFLAGS) -o semiprime_factor semiprime_factor.cpp $(LIBS)

test_suite: test_suite.cpp
	$(CXX) $(CXXFLAGS) -o test_suite test_suite.cpp $(LIBS)

clean:
	rm -f semiprime_factor test_suite

test: semiprime_factor
	@echo "Testing with known semiprimes..."
	./semiprime_factor 143
	./semiprime_factor 1234567890123456789
	./semiprime_factor 139789207152250802634791

.PHONY: all clean test