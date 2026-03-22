"""
flu/core/even_n.py
==================
Even-n Hyperprism construction via n = 2ᵏ · m decomposition.

ALGORITHM
─────────
For n = 2^k · m (m odd):
  1. micro[x] = g(x_0) ⊕ ... ⊕ g(x_{d-1})  (Gray-coded XOR over Z_{2^k})
     where g(x) = x ⊕ (x >> 1) is the binary reflected Gray map.
  2. macro[x] = (x_0 + x_1 + … + x_{d-1}) % m  (simple sum‑mod Latin hyperprism)
  3. combine: value = macro * 2^k + micro  (mixed‑radix Kronecker product)

THEOREM (Even‑n Latin Property), STATUS: PROVEN
─────────────────────────────────────────────────
  The combination of (a) Gray‑coded XOR micro‑block (bijection on Z_{2^k})
  and (b) sum‑mod macro‑block (bijection on Z_m) via mixed‑radix combination
  is a bijection of Z_n^D and preserves the Latin property (every axis‑aligned
  slice is a permutation of the digit set).  

  Proof sketch:
    • The micro‑array `_xor_latin` is Latin over Z_{2^k} because the XOR of
      Gray‑coded digits yields a permutation for any fixed axis.
    • The macro‑array `_sum_mod_latin` is Latin over Z_m by the same argument
      as the N‑ary theorem (T3): for any axis a, the sum of all coordinates
      sweeps all residues exactly once when x_a varies.
    • The Kronecker combination `macro * 2^k + micro` is a bijection on
      Z_n^d (mixed‑radix representation), and the Latin property follows
      because both components are Latin and the digits are independent.

  V15.2 Revision:
    - Macro coefficients simplified to all‑ones vector, guaranteeing Latin
      property for every odd m without special‑case failures.
    - XOR micro‑block provides Walsh‑spectral balance on the 2^k factor.
    - Vectorised contraction (`tensordot`) replaces ndindex loops.

Dependencies: numpy.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np


def decompose_n(n: int) -> Tuple[int, int]:
    """
    Split even n = 2^k * m (m odd).

    Returns
    -------
    k, m : int, int
    """
    k, m = 0, n
    while m % 2 == 0:
        m //= 2
        k += 1
    return k, m


def _sum_mod_latin(size: int, d: int) -> np.ndarray:
    """
    Latin hypercube over Z_size: cell = (x_0 + x_1 + … + x_{d-1}) % size.
    Always Latin for any size (proof: T3 / N‑ARY‑1).
    """
    if size == 1:
        return np.zeros([1] * d, dtype=np.int64)
    # All‑ones coefficients -> sum‑mod construction
    coeffs = np.ones(d, dtype=np.int64)
    grids = np.indices([size] * d, dtype=np.int64)
    # Vectorised contraction: Σ coeffs[a] * grids[a, ...]
    return np.tensordot(coeffs, grids, axes=(0, 0)) % size


def _xor_latin(size: int, d: int) -> np.ndarray:
    """
    Gray‑coded XOR over Z_{2^k}: cell = XOR_{a} g(x_a),
    where g(x) = x XOR (x >> 1) (binary reflected Gray code).
    This yields a Latin hypercube over the power‑of‑two modulus.
    """
    grids = np.indices([size] * d, dtype=np.int64)
    gray = grids ^ (grids >> 1)
    return np.bitwise_xor.reduce(gray, axis=0)


def generate(
    n: int, d: int, signed: bool = True, use_xor: bool = True
) -> np.ndarray:
    """
    Generate an n^d Latin hyperprism for even n.

    Parameters
    ----------
    n : int       even radix (must be even)
    d : int       spatial dimension
    signed : bool if True, centre the values to [‑floor(n/2), floor(n/2)-1]
    use_xor : bool if True (default), use Gray‑coded XOR for the 2^k factor;
                    if False, use sum‑mod for that factor as well (useful for
                    debugging or when XOR is not needed).

    Returns
    -------
    np.ndarray   shape (n,)*d with values in the appropriate range.
    """
    if n % 2 != 0:
        raise ValueError("generate() for even n requires an even argument.")
    k, m = decompose_n(n)
    step = 2 ** k

    # 1. Micro‑block over Z_{2^k}
    if use_xor:
        micro = _xor_latin(step, d)
    else:
        micro = _sum_mod_latin(step, d)

    # 2. Macro‑block over Z_m
    if m == 1:
        macro = np.zeros([1] * d, dtype=np.int64)
    else:
        macro = _sum_mod_latin(m, d)

    # 3. Mixed‑radix combination (Kronecker product)
    # macro * step + micro, with broadcasting via np.kron and np.tile
    hyperprism = (
        np.kron(macro, np.ones([step] * d, dtype=np.int64)) * step
        + np.tile(micro, [m] * d)
    )

    if signed:
        hyperprism -= n // 2
    return hyperprism


def verify(n: int, d: int) -> Dict[str, Any]:
    """
    Verify Latin property and coverage for the even‑n hyperprism.

    Returns
    -------
    dict with fields: n, d, shape, latin_ok, violations, coverage_ok,
                      min, max.
    """
    from flu.utils.verification import check_latin, check_coverage

    hp = generate(n, d, signed=False)
    latin_result = check_latin(hp, n, signed=False)
    coverage_result = check_coverage(hp, n, d, signed=False)

    return {
        "n": n,
        "d": d,
        "shape": hp.shape,
        "latin_ok": latin_result["latin_ok"],
        "violations": latin_result["violations"],
        "coverage_ok": coverage_result["coverage_ok"],
        "min": int(hp.min()),
        "max": int(hp.max()),
    }
