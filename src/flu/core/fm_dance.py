"""
flu/core/fm_dance.py
====================
FM-Dance ADDRESSING bijection: step index k ↔ signed d-tuple coordinates.

IMPORTANT: ADDRESSING vs. TRAVERSAL DISTINCTION
────────────────────────────────────────────────────────────────────
This module implements the ADDRESSING bijection:
    coord_i = (k // n^i) % n  − ⌊n/2⌋

This is the standard n-ary (base-n) representation with a centring shift.
It is a bijection and mean-centred, but it is NOT the kinetic traversal.

For the true FM-Dance kinetic (Hamiltonian) traversal that generalises the
de la Loubère Siamese algorithm — with proven step bound min(d, ⌊n/2⌋) and
Latin hypercube property — see:
    flu.core.fm_dance_path  (fm_dance_path.py)

Mathematical relationship:
    Addressing: coord_i = digit_i − half  (independent per digit, no Siamese)
    Kinetic:    x_i = prefix_sum of signed increments (coupled, Siamese)
    Both are bijections Z_n^d ↔ [0, n^d); only kinetic is Hamiltonian.

THEOREM (FM-Dance Addressing Bijection), STATUS: PROVEN — see inline.

The bijection is the standard n-ary (base-n) representation:
    k  =  Σᵢ digit_i · nⁱ,   digit_i ∈ [0, n)
    coord_i = digit_i − half,  half = ⌊n/2⌋

This is trivially a bijection, mean-centered, and Latin.

Only odd n supported.  Even-n → use core/even_n.py.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from flu.utils.math_helpers import is_odd


def index_to_coords(k: int, n: int, d: int) -> Tuple[int, ...]:
    """
    Step index k → signed d-tuple coordinate.

    THEOREM (Forward Bijection), STATUS: PROVEN
    Proof: k = Σᵢ dᵢ·nⁱ (unique n-ary representation); coord_i = dᵢ − half.
    Distinct k → distinct digit tuples → distinct coords.  □

    O(d) time.
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got {n}")
    total = n ** d
    if not (0 <= k < total):
        raise ValueError(f"k={k} out of range [0, {total})")
    half = n // 2
    digits = []
    rem = k
    for _ in range(d):
        digits.append(rem % n - half)
        rem //= n
    return tuple(digits)


def coords_to_index(coords: Tuple[int, ...], n: int, d: int) -> int:
    """
    Signed d-tuple coordinate → step index k.

    THEOREM (Inverse Bijection), STATUS: PROVEN
    Proof: k = Σᵢ (coord_i + half) · nⁱ — direct inversion of index_to_coords.  □

    O(d) time.
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got {n}")
    half = n // 2
    k = 0
    power = 1
    for c in coords:
        k += (c + half) * power
        power *= n
    return k


def generate_fast(
    n: int,
    d: int,
    signed: bool = True,
    start_pos: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Materialise the full n^d hyperprism.

    construct[i₁,...,i_d] = FM-Dance step index k whose coordinate is
    (i₁-half, …, i_d-half).

    THEOREM (Latin property of generate_fast), STATUS: PROVEN
    For a fixed slice along axis a, k = C + coord_a · n^a, varying coord_a
    over [0,n) gives n values differing by n^a — all distinct.  □

    O(n^d · d) time,  O(n^d) space.
    For sparse/high-d access, use index_to_coords directly.
    """
    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got {n}")
    total = n ** d
    half = n // 2
    construct = np.zeros([n] * d, dtype=np.int64)
    for k in range(total):
        coords = index_to_coords(k, n, d)
        idx = tuple(c + half for c in coords)
        construct[idx] = k
    return construct


def verify_bijection(n: int, d: int, verbose: bool = False) -> Dict:
    """Full round-trip verification for n^d."""
    total = n ** d
    errors = 0
    coords_all = []
    for k in range(total):
        coords = index_to_coords(k, n, d)
        k_back = coords_to_index(coords, n, d)
        coords_all.append(coords)
        if k_back != k:
            errors += 1
    coverage = len(set(coords_all)) == total
    arr = np.array(coords_all, dtype=float)
    mean_ok = bool(np.allclose(arr.mean(axis=0), 0.0, atol=1e-10))
    ok = errors == 0 and coverage and mean_ok
    if verbose:
        print(f"  n={n:2d}, d={d}: total={total}  mean={mean_ok}  {'✓' if ok else '✗'}")
    return {"n": n, "d": d, "total": total, "rt_errors": errors,
            "coverage": coverage, "mean_centered": mean_ok, "bijection_ok": ok}
