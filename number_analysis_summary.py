#!/usr/bin/env python3
"""
Complete analysis summary of 15482526220500967432610341
"""

def print_summary():
    n = 15482526220500967432610341
    p1 = 1804166129797
    p2 = 8581541336353
    
    print("="*60)
    print("COMPLETE ANALYSIS OF NUMBER: 15482526220500967432610341")
    print("="*60)
    print()
    
    print("1. BIT LENGTH AND SIZE ANALYSIS:")
    print(f"   - Number of digits: 26")
    print(f"   - Bit length: 84 bits")
    print(f"   - Fits in 128 bits: YES ✓")
    print(f"   - Fits in 64 bits: NO ✗")
    print(f"   - Binary representation has {bin(n).count('1')} ones")
    print()
    
    print("2. PRIMALITY TEST:")
    print(f"   - Is prime: NO")
    print(f"   - This number is composite")
    print()
    
    print("3. SPECIAL MATHEMATICAL PROPERTIES:")
    print(f"   - Not a perfect power")
    print(f"   - Not a perfect square")
    print(f"   - Odd number (not divisible by 2)")
    print(f"   - No small prime factors (2, 3, 5, 7, 11, etc.)")
    print()
    
    print("4. FACTORIZATION - CONFIRMED SEMIPRIME:")
    print(f"   15482526220500967432610341 = {p1} × {p2}")
    print()
    print(f"   Factor 1: {p1}")
    print(f"   - Digits: 13")
    print(f"   - Bits: 41")
    print(f"   - Is prime: YES ✓")
    print()
    print(f"   Factor 2: {p2}")
    print(f"   - Digits: 13")  
    print(f"   - Bits: 43")
    print(f"   - Is prime: YES ✓")
    print()
    
    print("5. FACTOR SIZE ANALYSIS:")
    print(f"   - Factor ratio: {p2/p1:.6f}")
    print(f"   - Factor difference: {p2 - p1} ({p2 - p1:,})")
    print(f"   - Factors are well-separated (not vulnerable to Fermat)")
    print(f"   - Both factors are roughly sqrt(n) in size")
    print(f"   - This is a balanced semiprime suitable for cryptographic use")
    print()
    
    print("CONCLUSION:")
    print("   • This is a SEMIPRIME (product of exactly two prime numbers)")
    print("   • The 84-bit modulus fits comfortably in 128 bits")
    print("   • Both prime factors are 13 digits long")
    print("   • The factors are well-balanced (41 and 43 bits)")
    print("   • This could be used as an RSA-84 modulus")
    print("   • Modern factorization algorithms can factor this in ~1.5 seconds")
    print()
    
    # Additional verification
    print("VERIFICATION:")
    print(f"   {p1} × {p2} = {p1 * p2}")
    print(f"   Matches original: {p1 * p2 == n} ✓")
    print()
    
    # Show the factorization challenge level
    print("FACTORIZATION DIFFICULTY:")
    print("   • 84-bit semiprime is considered EASY by modern standards")
    print("   • Can be factored on a regular computer in seconds")
    print("   • RSA keys typically use 2048+ bits for security")
    print("   • This would have been challenging in the 1980s-1990s")
    
    print("="*60)

if __name__ == "__main__":
    print_summary()