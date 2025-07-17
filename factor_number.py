#!/usr/bin/env python3
"""
More aggressive factorization attempt of 15482526220500967432610341
"""

from sympy import factorint, isprime
import time

def factor_with_timing(n):
    """Attempt to factor the number with timing"""
    print(f"Attempting to factor: {n}")
    print(f"Number has {len(str(n))} digits and {n.bit_length()} bits")
    print()
    
    # First quick primality check
    print("Checking primality...")
    start = time.time()
    is_prime = isprime(n)
    prime_time = time.time() - start
    print(f"Primality check took {prime_time:.3f} seconds")
    print(f"Is prime: {is_prime}")
    print()
    
    if not is_prime:
        print("Attempting factorization with sympy.factorint()...")
        print("This may take a while for large semiprimes...")
        
        start = time.time()
        try:
            # Use more aggressive settings
            factors = factorint(n, multiple=True, limit=1000000, use_trial=True, use_rho=True, use_pm1=True)
            factor_time = time.time() - start
            
            print(f"\nFactorization completed in {factor_time:.3f} seconds!")
            print(f"Factors: {factors}")
            
            # If it returns a list (multiple=True), convert to dict
            if isinstance(factors, list):
                factor_dict = {}
                for f in factors:
                    factor_dict[f] = factor_dict.get(f, 0) + 1
                factors = factor_dict
            
            # Analyze the factorization
            if len(factors) == 1 and n in factors:
                print("\nNo factors found - number might be prime or factorization is too hard")
            else:
                print("\nFactorization found!")
                total_factors = sum(factors.values())
                unique_factors = len(factors)
                
                print(f"Number of prime factors (with multiplicity): {total_factors}")
                print(f"Number of unique prime factors: {unique_factors}")
                
                # List factors in detail
                for prime, count in sorted(factors.items()):
                    print(f"  {prime} (appears {count} time{'s' if count > 1 else ''})")
                    print(f"    - {len(str(prime))} digits")
                    print(f"    - {prime.bit_length()} bits")
                    print(f"    - Is prime: {isprime(prime)}")
                
                # Verify factorization
                product = 1
                for prime, count in factors.items():
                    product *= prime ** count
                
                print(f"\nVerification: Product of factors = {product}")
                print(f"Matches original number: {product == n}")
                
                if total_factors == 2 and unique_factors == 2:
                    print("\n*** This is a SEMIPRIME (product of exactly two primes)! ***")
                elif total_factors == 2 and unique_factors == 1:
                    print("\n*** This is a square of a prime! ***")
                
        except KeyboardInterrupt:
            print("\nFactorization interrupted by user")
        except Exception as e:
            print(f"\nError during factorization: {e}")

# Main execution
if __name__ == "__main__":
    number = 15482526220500967432610341
    factor_with_timing(number)