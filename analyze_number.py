#!/usr/bin/env python3
"""
Analyze the mathematical properties of the number 15482526220500967432610341
"""

import math
from sympy import factorint, isprime, primefactors
import random

def analyze_number(n):
    """Comprehensive analysis of a number's mathematical properties"""
    print(f"Analyzing number: {n}")
    print(f"Number of digits: {len(str(n))}")
    print()
    
    # 1. Calculate bit length
    bit_length = n.bit_length()
    print(f"1. Bit length: {bit_length} bits")
    print(f"   Fits in 128 bits: {'Yes' if bit_length <= 128 else 'No'}")
    print(f"   Minimum bits needed: {bit_length}")
    print()
    
    # 2. Primality test
    print("2. Primality test:")
    is_prime = isprime(n)
    print(f"   Is prime: {is_prime}")
    
    if not is_prime:
        # Quick trial division for small factors
        small_factors = []
        temp_n = n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while temp_n % p == 0:
                small_factors.append(p)
                temp_n //= p
        
        if small_factors:
            print(f"   Small prime factors found: {small_factors}")
            print(f"   Remaining number after division: {temp_n}")
    print()
    
    # 3. Special mathematical properties
    print("3. Special mathematical properties:")
    
    # Check if it's a perfect power
    for k in range(2, int(math.log2(n)) + 1):
        root = int(n ** (1/k))
        if root ** k == n:
            print(f"   Perfect {k}th power: {root}^{k}")
            break
    else:
        print("   Not a perfect power")
    
    # Check if it's a square
    sqrt_n = int(math.sqrt(n))
    if sqrt_n * sqrt_n == n:
        print(f"   Perfect square: {sqrt_n}²")
    
    # Check some modular properties
    print(f"   n mod 2 = {n % 2} ({'even' if n % 2 == 0 else 'odd'})")
    print(f"   n mod 3 = {n % 3}")
    print(f"   n mod 5 = {n % 5}")
    print(f"   n mod 7 = {n % 7}")
    print()
    
    # 4. Factorization attempt
    print("4. Factorization analysis:")
    
    # First do trial division with small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    remaining = n
    small_factors = {}
    
    for p in small_primes:
        count = 0
        while remaining % p == 0:
            count += 1
            remaining //= p
        if count > 0:
            small_factors[p] = count
    
    if small_factors:
        print(f"   Small factors found: {small_factors}")
        print(f"   Remaining number: {remaining}")
        
        # Check if remaining is prime
        if remaining > 1 and isprime(remaining):
            print(f"   Remaining number is prime!")
            small_factors[remaining] = 1
    else:
        print("   No small prime factors found")
    
    # If no small factors, try more advanced factorization
    if not small_factors and n > 1:
        print("   Attempting advanced factorization...")
        
        # Try Pollard's rho method
        def pollard_rho(n, max_iterations=10000):
            if n % 2 == 0:
                return 2
            
            x = 2
            y = 2
            d = 1
            
            def f(x):
                return (x * x + 1) % n
            
            iterations = 0
            while d == 1 and iterations < max_iterations:
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), n)
                iterations += 1
            
            if d != n and d != 1:
                return d
            return None
        
        factor = pollard_rho(n)
        if factor:
            other_factor = n // factor
            print(f"   Found factorization using Pollard's rho!")
            print(f"   {n} = {factor} × {other_factor}")
            
            # Check if factors are prime
            if isprime(factor):
                print(f"   First factor {factor} is prime")
            if isprime(other_factor):
                print(f"   Second factor {other_factor} is prime")
            
            if isprime(factor) and isprime(other_factor):
                print("   This is a semiprime (product of two primes)!")
                print(f"   Factor 1: {factor} ({len(str(factor))} digits, {factor.bit_length()} bits)")
                print(f"   Factor 2: {other_factor} ({len(str(other_factor))} digits, {other_factor.bit_length()} bits)")
        else:
            print("   Pollard's rho didn't find factors quickly")
            print("   This might be a prime or a hard semiprime")
    
    print()
    
    # 5. Estimate factor sizes for semiprime
    print("5. Factor size estimation (assuming semiprime):")
    
    # If it's a semiprime p*q, estimate factor sizes
    sqrt_n = math.isqrt(n)
    print(f"   Square root: ~{sqrt_n}")
    print(f"   If semiprime with balanced factors:")
    print(f"     Each factor would be approximately {len(str(sqrt_n))} digits")
    print(f"     Each factor would need approximately {sqrt_n.bit_length()} bits")
    
    # Check if factors might be close (vulnerable to Fermat factorization)
    print(f"\n   Checking for close factors (Fermat factorization attempt):")
    a = math.isqrt(n) + 1
    attempts = 0
    max_attempts = 10000
    
    while attempts < max_attempts:
        b_squared = a * a - n
        b = math.isqrt(b_squared)
        
        if b * b == b_squared:
            p = a - b
            q = a + b
            print(f"   Found factors using Fermat's method!")
            print(f"   p = {p}")
            print(f"   q = {q}")
            print(f"   Verification: p × q = {p * q}")
            print(f"   Factor difference: {abs(p - q)}")
            break
        
        a += 1
        attempts += 1
    else:
        print(f"   No close factors found in {max_attempts} attempts")
        print("   Factors are likely well-separated")
    
    # Additional analysis
    print(f"\n   Additional semiprime analysis:")
    print(f"   If this is RSA-like (product of two similar-sized primes):")
    print(f"     Bit size of modulus: {bit_length}")
    print(f"     Estimated bit size of each prime: ~{bit_length // 2}")
    
    # Check if it could be an RSA modulus
    if bit_length % 2 == 0:
        print(f"     Could be RSA-{bit_length} modulus")
    
    return n

# Main execution
if __name__ == "__main__":
    number = 15482526220500967432610341
    analyze_number(number)