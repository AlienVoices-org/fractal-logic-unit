"""
flu/utils/math_helpers.py
=========================
Pure mathematical helpers shared across the package.

Single Responsibility: fast, cached arithmetic primitives.
No package-internal imports.

Provides
--------
factorial(n)           – cached int factorial
inv_mod(a, m)          – modular multiplicative inverse (for prime m)
is_odd(n)              – simple parity test with helpful error on n<1
digits_signed(n)       – balanced digit set for base-n
digits_unsigned(n)     – unsigned digit set for base-n
mean_of_digits(n, ...) – theoretical mean of the digit set
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List


# ── Factorial ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def factorial(n: int) -> int:
    """
    Cached integer factorial.

    Parameters
    ----------
    n : int  (≥ 0)

    Returns
    -------
    n! : int

    Raises
    ------
    ValueError  if n < 0
    """
    if n < 0:
        raise ValueError(f"factorial requires n≥0, got {n}")
    return math.factorial(n)


# ── Modular inverse ───────────────────────────────────────────────────────────

def inv_mod(a: int, m: int) -> int:
    """
    Modular multiplicative inverse: a · inv_mod(a,m) ≡ 1  (mod m).

    Uses Python's built-in pow(a, -1, m) which requires m to be prime
    or gcd(a,m)=1.  FM-Dance only ever calls this with prime n (3,5,7,…),
    so the precondition always holds in practice.

    Parameters
    ----------
    a : int   (non-zero mod m)
    m : int   (> 1, gcd(a,m) = 1)

    Returns
    -------
    int   in [1, m-1]

    Raises
    ------
    ValueError  if gcd(a, m) ≠ 1
    """
    g = math.gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No inverse: gcd({a},{m})={g}")
    return pow(int(a), -1, int(m))


# ── Parity ────────────────────────────────────────────────────────────────────

def is_odd(n: int) -> bool:
    """Return True iff n is odd.  Raises ValueError for n < 1."""
    if n < 1:
        raise ValueError(f"Expected n≥1, got {n}")
    return n % 2 == 1


# ── Digit sets ────────────────────────────────────────────────────────────────

def digits_signed(n: int) -> List[int]:
    """
    Balanced (signed) digit set for base n.

    Odd  n → {-(n-1)//2, …, 0, …, (n-1)//2}   mean = 0 exactly.
    Even n → {-n//2+1,   …, 0, …,  n//2}       mean = 0.5.

    STATUS: PROVEN  (mean follows from symmetric sum for odd n;
                     near-symmetry for even n is by construction)
    """
    if n < 2:
        raise ValueError(f"n must be ≥ 2, got {n}")
    if n % 2 == 1:
        half = n // 2
        return list(range(-half, half + 1))
    else:
        half = n // 2
        return list(range(-half + 1, half + 1))


def digits_unsigned(n: int) -> List[int]:
    """Unsigned digit set {0, 1, …, n-1} for base n."""
    if n < 2:
        raise ValueError(f"n must be ≥ 2, got {n}")
    return list(range(n))


def mean_of_digits(n: int, signed: bool = True) -> float:
    """
    Theoretical mean of the digit set for base n.

    STATUS: PROVEN
      signed odd  n → 0.0   (symmetric set, sum = 0)
      signed even n → 0.5   (near-symmetric, off by 1/n)
      unsigned    n → (n-1)/2
    """
    if signed:
        return 0.0 if (n % 2 == 1) else 0.5
    return (n - 1) / 2
