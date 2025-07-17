#!/usr/bin/env python3

import math

def find_factors(n):
    """Find factors of n using trial division"""
    if n <= 1:
        return []
    
    factors = []
    # Check for factor of 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Check odd factors up to sqrt(n)
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    
    if n > 2:
        factors.append(n)
    
    return factors

# Test cases that need correction
numbers = [
    15241383247,
    8776260683437,
    123456789012345678901,
]

print("Finding correct factors...")
for num in numbers:
    print(f"\nNumber: {num}")
    factors = find_factors(num)
    print(f"Prime factors: {factors}")
    
    if len(factors) == 2:
        print(f"Semiprime: {factors[0]} × {factors[1]} = {num}")
    elif len(factors) == 1:
        print("This is a prime number")
    else:
        print(f"Has {len(factors)} prime factors")
        
    # Verify
    product = 1
    for f in factors:
        product *= f
    print(f"Verification: {product} == {num}: {product == num}")