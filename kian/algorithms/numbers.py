"""Number theory and basic combinatorics: gcd/lcm, the Sieve of Eratosthenes,
prime factorisation, modular exponentiation, and the extended Euclid."""

from __future__ import annotations

from typing import Dict, List


def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def extended_gcd(a: int, b: int):
    """Return (g, x, y) with a*x + b*y = g = gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def sieve(n: int) -> List[int]:
    """All primes <= n via the Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    i = 2
    while i * i <= n:
        if is_p[i]:
            is_p[i * i :: i] = b"\x00" * len(is_p[i * i :: i])
        i += 1
    return [i for i in range(2, n + 1) if is_p[i]]


def prime_factors(n: int) -> Dict[int, int]:
    """Factorise ``n`` into {prime: exponent}."""
    n = abs(n)
    out: Dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def mod_pow(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation: base**exp % mod in O(log exp)."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result
