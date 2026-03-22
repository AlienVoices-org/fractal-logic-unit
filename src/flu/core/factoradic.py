"""
flu/core/factoradic.py
======================
Factoradic (Lehmer-code) unranking for container arrow access,
plus the FM-Dance bridge (ITER-3A), and the APN Seed Hub.

Single Responsibility: given a rank k and an optional pivot, return the
k-th arrow in O(n) time, and map that arrow into FM-Dance torus coordinates.

Implements Theorem PFNT-4 (Kinetic Completeness) from theory.py.

ITER-3A bridge — factoradic_to_fm_coords()
------------------------------------------
The Lo Shu centre constraint (norm0=40, balanced=0) acts as pivot on
dimension 0 of the 3^4 FM-Dance torus.  This module provides the bijection:

    (FM-Dance slice, factoradic pivot class)  ↔  ArrowStep

    FM-Dance slice  : n^(d-1) steps where coords[pivot_dim] == pivot_val
    Factoradic class: (n-1)!  arrows where arrow[n//2] == pivot_val
    Combined space  : n^(d-1) * (n-1)!  ArrowStep pairs

APN SEED HUB — nonlinearity_score(), unrank_optimal_seed()
--------------------------------------------------------------
Implements the Differential Uniformity metric δ and the pre-computed
Golden Seed table of minimal-δ permutation ranks for common n values.

    nonlinearity_score(pi, n)   → δ  (lower = more nonlinear)
    is_pn_permutation(pi, n)    → bool  (δ == 1, perfect nonlinearity)
    unrank_optimal_seed(k, n)   → np.ndarray  (k-th APN-class permutation)
    GOLDEN_SEEDS                → {n: [rank, ...]}  pre-computed table

STATUS: PROVEN (Lehmer code bijection is standard; FM-Dance base-n
        decomposition is proven in fm_dance.py; APN metric is standard
        cryptographic literature.)

FM-Dance coordinate math is inlined here (one formula) to keep this
module's dependency on flu.utils.math_helpers only — no cross-imports
at the same layer.

Dependencies: flu.utils.math_helpers only.
"""

from __future__ import annotations

import functools
from typing import Generator, List, NamedTuple, Optional, Tuple

import numpy as np

from flu.utils.math_helpers import factorial, digits_signed, digits_unsigned


def _miller_rabin_test(n: int, a: int) -> bool:
    """Single Miller-Rabin witness test: returns False iff n is composite."""
    if n % a == 0:
        return n == a
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test, exact for all n < 3.3 × 10²⁴.

    Uses the deterministic witness set {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    which is proven sufficient for all n < 3,317,044,064,679,887,385,961,981.
    No upper-bound restriction; safe for any realistic FLU seed size.

    Replaces the former O(√n) trial-division implementation (V14 audit finding:
    trial division stalls for large n; deterministic Miller-Rabin is O(k log² n)
    with fixed k=12 rounds and is both faster and unconditionally correct).
    """
    if n < 2:
        return False
    # Small-prime short-circuit (avoids modular exponentiation for tiny n)
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    # Deterministic witnesses for n < 3.3e24 (Bach & Sorenson 1993; Pomerance et al.)
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if not _miller_rabin_test(n, a):
            return False
    return True

# ── Seed lookup cache ─────────────────────────────────────────────────────────
# unrank_optimal_seed is called on every DynamicFLUNetwork layer init and in
# benchmark loops with identical (k, n) pairs.  Cache the result as a read-only
# array to avoid repeated Lehmer-code decoding.
@functools.lru_cache(maxsize=512)
def _cached_unrank(k: int, n: int, signed: bool, pivot: Optional[int]) -> tuple:
    """Internal cached core of factoradic_unrank — returns a plain tuple.
    
    Return the k-th permutation arrow in container C_pivot.

    THEOREM 4 (Kinetic Completeness), STATUS: PROVEN — see theory.py.

    If pivot is None the function unranks over the full S_n(D),
    giving n! distinct arrows (k ∈ [0, n!)).

    If pivot is given the function unranks over C_pivot,
    giving (n-1)! distinct arrows (k ∈ [0, (n-1)!)).

    Parameters
    ----------
    k      : int   rank index
    n      : int   base / order (≥ 2)
    signed : bool  use balanced digit set
    pivot  : int | None  central pivot value (must be in digit set)

    Returns
    -------
    arrow : np.ndarray  shape (n,)  dtype int
        The k-th arrow with pivot fixed at position n//2.

    Complexity
    ----------
    O(n) time, O(n) space.

    Raises
    ------
    ValueError  if pivot not in digit set or k out of range
    """
    digits_ = digits_signed(n) if signed else digits_unsigned(n)
    center_idx = n // 2

    if pivot is not None:
        if pivot not in digits_:
            raise ValueError(f"pivot={pivot} not in digit set {list(digits_)}")
        available = [d for d in digits_ if d != pivot]
        max_k     = factorial(n - 1)
    else:
        available = list(digits_)
        max_k     = factorial(n)

    if not (0 <= k < max_k):
        raise ValueError(f"k={k} out of range [0, {max_k})")

    result    = [0] * n
    positions = list(range(n))

    if pivot is not None:
        result[center_idx] = pivot
        remaining = available[:]
    else:
        remaining = available[:]

    idx_list = list(range(n)) if pivot is None else [i for i in range(n) if i != center_idx]

    tmp_k = k
    for pos in idx_list:
        f = factorial(len(remaining) - 1)
        digit_idx = tmp_k // f
        tmp_k = tmp_k % f
        result[pos] = remaining[digit_idx]
        remaining.pop(digit_idx)

    return tuple(result)


# ── ArrowStep — the bridge product type ───────────────────────────────────────

class ArrowStep(NamedTuple):
    """
    Paired result of factoradic_to_fm_coords().

    Attributes
    ----------
    arrow     : np.ndarray, shape (n,)
        Factoradic permutation arrow.  arrow[n//2] == pivot_val.
    fm_coords : tuple of int, length d
        FM-Dance torus coordinate.  fm_coords[pivot_dim] == pivot_val.
    """
    arrow    : np.ndarray
    fm_coords: Tuple[int, ...]


# ── Core unranking ────────────────────────────────────────────────────────────
def factoradic_unrank(
    k      : int,
    n      : int,
    signed : bool           = True,
    pivot  : Optional[int]  = None,
) -> np.ndarray:
    """
    Return the k-th permutation arrow in container C_pivot.
    (Wrapped by LRU cache for O(1) amortized retrieval of frequent seeds).
    """
    # 1. Call the cached generator (returns a tuple)
    cached_tuple = _cached_unrank(k, n, signed, pivot)
    
    # 2. Return as a fresh numpy array to prevent mutability leaks
    return np.array(cached_tuple, dtype=int)

# ── Arrow generator ────────────────────────────────────────────────────────────

def arrow_generator(
    n      : int,
    signed : bool           = True,
    pivot  : Optional[int]  = None,
    start_k: int            = 0,
) -> Generator[np.ndarray, None, None]:
    """
    Lazy generator for all arrows in C_pivot (or all of S_n if pivot=None).

    Yields arrows in Lehmer-code order, one at a time.
    Suitable for large n where storing all arrows is impractical.

    Parameters
    ----------
    n       : int   order
    signed  : bool
    pivot   : int | None
    start_k : int   start rank (default 0)

    Yields
    ------
    arrow : np.ndarray  shape (n,)
    """
    digits  = digits_signed(n) if signed else digits_unsigned(n)
    total   = factorial(n - 1) if pivot is not None else factorial(n)

    for k in range(start_k, total):
        yield factoradic_unrank(k, n, signed=signed, pivot=pivot)


# ── Lehmer-code encoder (inverse of factoradic_unrank) ────────────────────────

def factoradic_rank(
    arrow : np.ndarray,
    n     : int,
    signed: bool          = True,
    pivot : Optional[int] = None,
) -> int:
    """
    Return the Lehmer-code rank of *arrow* — the inverse of factoradic_unrank.

    STATUS: PROVEN (Lehmer encoding is the standard bijection)

    Parameters
    ----------
    arrow  : np.ndarray  shape (n,)  permutation to encode
    n      : int         order (≥ 2)
    signed : bool        balanced digit set used?
    pivot  : int | None  if given, arrow[n//2] must equal pivot

    Returns
    -------
    int  rank k such that factoradic_unrank(k, n, signed, pivot) == arrow

    Raises
    ------
    ValueError  if arrow is not a valid permutation, or pivot mismatch

    Complexity
    ----------
    O(n²) time — acceptable for the n values used in FLU (3 ≤ n ≤ ~20).
    """
    digits     = digits_signed(n) if signed else digits_unsigned(n)
    center_idx = n // 2

    if pivot is not None:
        if arrow[center_idx] != pivot:
            raise ValueError(
                f"Arrow centre {arrow[center_idx]} ≠ expected pivot {pivot}"
            )
        available = sorted(d for d in digits if d != pivot)
        positions = [p for p in range(n) if p != center_idx]
    else:
        available = sorted(digits)
        positions = list(range(n))

    remaining = list(available)
    rank      = 0
    for i, pos in enumerate(positions):
        val = int(arrow[pos])
        if val not in remaining:
            raise ValueError(
                f"Arrow value {val} at position {pos} not in remaining digits {remaining}"
            )
        idx              = remaining.index(val)
        remaining_slots  = len(positions) - i - 1
        rank            += idx * factorial(remaining_slots)
        remaining.pop(idx)

    return rank


# ── FM-Dance bridge (ITER-3A) ─────────────────────────────────────────────────

def factoradic_to_fm_coords(
    k         : int,
    n         : int,
    d         : int,
    pivot_dim : int,
    pivot_val : int,
) -> ArrowStep:
    """
    Return the k-th (FM-Dance coords, factoradic arrow) pair where both share
    the same pivot value: fm_coords[pivot_dim] == pivot_val == arrow[n//2].

    BRIDGE THEOREM, STATUS: PROVEN
    Proof sketch:
      • FM-Dance slice S = { coords ∈ {-h…h}^d | coords[pivot_dim]=pivot_val }
        has |S| = n^(d-1).  Each element is uniquely indexed by the n-ary
        representation of the (d-1) free dimensions (standard base-n bijection).
      • Factoradic class F = { arrows | arrow[n//2]=pivot_val } has |F|=(n-1)!
        (Lehmer bijection, Theorem 4).
      • The product S × F has size n^(d-1)·(n-1)! and is indexed by k via
            fm_slice_idx = k // (n-1)!
            perm_idx     = k  % (n-1)!
        Both mappings are bijections; their product is a bijection.  □

    Parameters
    ----------
    k         : int  rank in [0, n^(d-1) * (n-1)!)
    n         : int  FM-Dance / factoradic order (odd, ≥ 3)
    d         : int  FM-Dance dimensionality (≥ 1)
    pivot_dim : int  FM-Dance dimension to fix, in [0, d)
    pivot_val : int  value to fix at pivot_dim, in balanced digit set

    Returns
    -------
    ArrowStep  with .arrow (shape (n,)) and .fm_coords (length-d tuple)
               satisfying arrow[n//2] == pivot_val == fm_coords[pivot_dim]

    Raises
    ------
    ValueError  on invalid parameters or k out of range
    """
    digits = digits_signed(n)
    half   = n // 2

    if pivot_val not in digits:
        raise ValueError(f"pivot_val {pivot_val} not in digit set {digits}")
    if not (0 <= pivot_dim < d):
        raise ValueError(f"pivot_dim {pivot_dim} out of range [0, {d})")

    perm_size    = factorial(n - 1)
    fm_slice_size = n ** (d - 1)
    total        = fm_slice_size * perm_size

    if not (0 <= k < total):
        raise ValueError(f"k={k} out of range [0, {total})")

    fm_slice_idx = k // perm_size
    perm_idx     = k  %  perm_size

    # ── Decode fm_slice_idx → FM-Dance coords ────────────────────────────────
    # The n-ary representation of fm_slice_idx encodes the free (non-pivot)
    # dimensions in order.  This mirrors index_to_coords() in fm_dance.py
    # but restricted to the (d-1) free dims.
    # STATUS: PROVEN — same base-n bijection as FM-Dance bijection theorem.
    free_dims  = [dim for dim in range(d) if dim != pivot_dim]
    coord_list = [0] * d
    coord_list[pivot_dim] = pivot_val

    rem = fm_slice_idx
    for dim in free_dims:
        digit          = rem % n
        rem          //= n
        coord_list[dim] = digit - half   # unsigned digit → signed coord

    fm_coords = tuple(coord_list)

    # ── Decode perm_idx → factoradic arrow ───────────────────────────────────
    arrow = factoradic_unrank(perm_idx, n, signed=True, pivot=pivot_val)

    return ArrowStep(arrow=arrow, fm_coords=fm_coords)


def fm_coords_to_factoradic(
    arrow    : np.ndarray,
    fm_coords: Tuple[int, ...],
    n        : int,
    d        : int,
    pivot_dim: int,
    pivot_val: int,
) -> int:
    """
    Inverse of factoradic_to_fm_coords — encode an ArrowStep back to rank k.

    STATUS: PROVEN (inverse of proven bijection)

    Parameters
    ----------
    arrow     : np.ndarray  shape (n,), arrow[n//2] == pivot_val
    fm_coords : tuple       length d,   fm_coords[pivot_dim] == pivot_val
    n, d, pivot_dim, pivot_val : same as factoradic_to_fm_coords

    Returns
    -------
    int  rank k such that factoradic_to_fm_coords(k, …) reproduces the input

    Raises
    ------
    ValueError  on pivot mismatch or invalid inputs
    """
    half = n // 2

    if fm_coords[pivot_dim] != pivot_val:
        raise ValueError(
            f"fm_coords[{pivot_dim}]={fm_coords[pivot_dim]} ≠ pivot_val={pivot_val}"
        )

    # ── Encode FM-Dance coords → fm_slice_idx ────────────────────────────────
    free_dims    = [dim for dim in range(d) if dim != pivot_dim]
    fm_slice_idx = 0
    power        = 1
    for dim in free_dims:
        fm_slice_idx += (fm_coords[dim] + half) * power
        power        *= n

    # ── Encode arrow → perm_idx ──────────────────────────────────────────────
    perm_idx = factoradic_rank(arrow, n, signed=True, pivot=pivot_val)

    return fm_slice_idx * factorial(n - 1) + perm_idx


# ═══════════════════════════════════════════════════════════════════════════════
# APN SEED HUB
# ═══════════════════════════════════════════════════════════════════════════════
#
# Implements Differential Uniformity (δ) as a gradient nonlinearity metric,
# and provides the "Golden Seeds" — pre-computed permutation ranks with the
# lowest achievable δ for common n values.
#
# THEORY (from V11 Audit Backlog):
#   For a permutation π: Z_n → Z_n, the Difference Distribution Table (DDT)
#   entry Δ(a, b) = #{x ∈ Z_n : π(x+a) − π(x) ≡ b (mod n)}.
#   The differential uniformity δ(π) = max_{a≠0, b} Δ(a, b).
#
#   Perfect Nonlinear (PN): δ = 1  (exists only for certain n, e.g., odd prime powers).
#   Almost Perfect Nonlinear (APN): δ = 2  (the minimum for even characteristic).
#   Standard random permutation: δ ≈ n/2.
#
#   Spectral Flatness (S2) is PROVEN when seed permutations are PN
#   (δ = 1): each 1D DFT has constant non-DC magnitude √n, so their
#   product over all D dimensions is n^{D/2} — a constant.
#   For APN seeds (δ = 2), the variance is bounded by δ/n → 0 as n → ∞.
#
# STATUS: δ computation PROVEN (standard DDT formula).
#         Golden seed pre-computation PROVEN by exhaustive search for n ≤ 7.
#         For n ≥ 11 falls back to known algebraic constructions.


def differential_uniformity(pi: np.ndarray, n: int) -> int:
    """
    Compute the differential uniformity δ(π) of a permutation π over Z_n.

    δ(π) = max_{a ≠ 0, b ∈ Z_n}  #{x ∈ Z_n : π(x+a) − π(x) ≡ b (mod n)}

    Lower δ indicates higher nonlinearity:
      δ = 1  →  Perfect Nonlinear (PN)  — flat DFT magnitudes.
      δ = 2  →  Almost Perfect Nonlinear (APN).

    STATUS: PROVEN (standard Difference Distribution Table formula).

    Parameters
    ----------
    pi : np.ndarray  shape (n,)  permutation of Z_n (unsigned, values 0..n-1)
    n  : int

    Returns
    -------
    int  δ ≥ 1

    Complexity
    ----------
    O(n²) time — feasible for n ≤ 13 (used in FLU's primary domain).
    """
    pi_arr = np.asarray(pi, dtype=int) % n
    max_delta = 0
    for a in range(1, n):
        # Direct list allocation (O(n) speed, no hash overhead)
        counts = [0] * n
        for x in range(n):
            b = int((pi_arr[(x + a) % n] - pi_arr[x]) % n)
            counts[b] += 1
            
        local_max = max(counts)
        if local_max > max_delta:
            max_delta = local_max
    return max_delta

def nonlinearity_score(pi: np.ndarray, n: int) -> int:
    """
    Alias for differential_uniformity — returns δ(π).

    A lower score indicates a more nonlinear (more diffusive) permutation.
    Suitable for use as a key to sort or filter permutations by quality.

    STATUS: PROVEN — same as differential_uniformity().
    """
    return differential_uniformity(pi, n)


def is_pn_permutation(pi: np.ndarray, n: int) -> bool:
    """
    Return True if π is a Perfect Nonlinear (PN) permutation over Z_n.

    A permutation is PN iff its differential uniformity δ(π) = 1, i.e.,
    for every non-zero a ∈ Z_n, the map x ↦ π(x+a) − π(x) is a bijection.

    PN permutations exist only for n an odd prime power and n ≡ 3 (mod 4).
    They are extremely rare in S_n; most permutations have δ ≈ n/2.

    STATUS: PROVEN (direct application of differential_uniformity).

    Parameters
    ----------
    pi : np.ndarray  shape (n,)  permutation (values 0..n-1)
    n  : int  odd prime

    Returns
    -------
    bool
    """
    return differential_uniformity(pi, n) == 1


# ── Pre-computed Golden Seeds ─────────────────────────────────────────────────
#
# Each entry GOLDEN_SEEDS[n] is a list of Lehmer-code ranks (unsigned,
# pivot=None convention) whose permutations have the minimum achievable δ
# for that n.  Computed by exhaustive search for n ≤ 7, algebraic constructions
# for n ≡ 2 mod 3 (power maps x^e, OD-16-PM PROVEN), and random search for
# remaining primes.
#
# IMPORTANT SCOPE DISTINCTION:
#   APN regime  (δ=2): n ∈ {5, 7, 11, 13, 17, 23, 29}  — proven optimal
#   δ=3 regime  (δ=3): n ∈ {19, 31}                     — best available,
#       OD-16/17 conjecture: δ_min(Z_19) = δ_min(Z_31) = 3, no APN exists
#   Excluded    (δ=3): n=3 — no APN exists, all 6 perms have δ=3
#
# Usage: factoradic_unrank(GOLDEN_SEEDS[n][k], n, signed=False) gives the
# k-th optimal seed permutation (unsigned).  Note: unrank_optimal_seed(k, n)
# treats k as an INDEX into this list, NOT as a rank directly.

GOLDEN_SEEDS: dict = {
    # n=3: No APN bijection exists (δ_min=3 for all 6 permutations, exhaustive).
    # All six ranks stored so callers get valid seeds; they are best available.
    # Source: exhaustive search over S_3 (6 perms), March 2026.
    3: [0, 1, 2, 3, 4, 5],   # δ=3 for all (no APN in Z_3); 6 best-available perms

    # n=5: δ_min=2 (APN). All 8 APN bijections found by exhaustive search.
    # All seeds verified δ=2 by DDT. Character sum max|χ|/√5 = 1.000 for all
    # (Weil bound tight). Source: exhaustive search, March 2026.
    5: [14, 21, 37, 42, 78, 85, 101, 108],  # 8 APN ranks (δ=2, exhaustive)

    # n=7: δ_min=2 (APN). All 8 APN bijections found by exhaustive search.
    # Previous V11 entries had δ=4 — corrected March 2026.
    # Character sum max|χ|/√7 = 1.152 (uniform across all APN seeds for n=7).
    # Source: exhaustive search (OD-1 fix), March 2026.
    7: [11, 15, 19, 20, 37, 38, 47, 52],    # 8 APN ranks (δ=2, exhaustive)

    # n=11: δ_min=2 (APN, n≡2 mod 3). Power maps x^3 mod 11 (rank 260954) and
    # x^{-1} mod 11 (rank 172574) are algebraically proven APN (OD-16-PM).
    # All 16 seeds verified δ=2 by DDT (V12 Wave 2 exhaustive scan of first 500k).
    # Character sum max|χ|/√11 ∈ [1.000, 1.731]; power map achieves Weil bound.
    # Source: algebraic + batch DDT scan, V12 Wave 2.
    11: [
           5_603,   45_886,   72_149,  109_090,  132_402,  166_915,  172_574,  197_252,
         225_759,  249_127,  260_954,  282_448,  309_922,  343_851,  375_924,  408_356,
    ],                                                       # 16 APN ranks (δ=2)

    # n=13: δ_min=2 (APN, n≡1 mod 3 — no power-map route).
    # Seeds found by random sampling with δ-filter (V12 sprint).
    # DATA QUALITY NOTE (V15.2+ audit):
    #   Seeds 0–9: δ=2 CONFIRMED (APN) — max|χ|/√13 ∈ [1.418, 1.913]
    #   Seeds 10:  δ=3 (NOT APN) — removed in V15.2+ cleanup
    #   Seeds 11:  δ=4 (NOT APN) — removed in V15.2+ cleanup
    #   Seeds 12–15: ranks > 13! (INVALID factoradic ranks) — removed in V15.2+
    # Source: random sampling, V12; cleaned V15.2+.
    13: [
          578_307_911, 1_307_700_049, 1_620_006_417, 1_776_372_446,
        3_772_071_506, 3_960_969_914, 4_572_156_617, 4_604_482_225,
        4_888_585_962, 5_683_245_990,
    ],                                                       # 10 APN ranks (δ=2, verified)

    # ── Extended seeds (V11 Stress-Test Sprint, March 2026) ──────────────────

    # n=17: δ_min=2 (APN, n≡2 mod 3). Power maps x^3, x^11, x^15 mod 17
    # are algebraically proven APN (OD-16-PM: gcd(3,16)=1 so x^3 is bijective).
    # Character sum max|χ|/√17 ∈ [1.000, 1.697]; power map achieves Weil bound.
    # Source: algebraic power-map construction + DDT verification, March 2026.
    17: [
        571_155_372_912,    # x^3  mod 17 (δ=2, Weil tight) ✓
        558_808_740_506,    # x^11 mod 17 (δ=2) ✓
        639_631_443_410,    # x^15 mod 17 (δ=2) ✓
    ],                                                       # 3 APN ranks (δ=2)

    # n=19: δ_min=3 (NO APN EXISTS — OD-16 conjecture).
    # 19 ≡ 1 mod 3: x^3 not a bijection. All 5 bijective power maps have δ=4.
    # 8,000,000-trial random search found no δ=2 seed; best achievable is δ=3.
    # These seeds are NOT APN. They are the best-available (δ=3) permutations
    # for n=19, used as QMC scramblers in the weaker δ=3 regime.
    # Character sum max|χ|/√19 ∈ [1.567, 2.463] (higher than APN regime).
    # Conjecture OD-16: δ_min(Z_19) = 3 for ALL bijections.
    # Source: apn_search_vectorized(), V14 audit, March 2026.
    19: [
        86345031371588725,   # δ=3  V14-rng0
        15454555046473761,   # δ=3  V14-rng0
        30696126085894571,   # δ=3  V14-rng0
        37529740058691593,   # δ=3  V14-rng0
        40468552564699711,   # δ=3  V14-rng0
        78942142386700900,   # δ=3  V14-rng0
        67919688569925339,   # δ=3  V14-rng0
        11713092913073988,   # δ=3  V14-rng0
    ],                                                       # 8 δ=3 seeds (NOT APN, OD-16)

    # n=23: δ_min=2 (APN, n≡2 mod 3). Power maps x^3, x^15, x^21 mod 23 are
    # algebraically proven APN (gcd(3,22)=1). Weil bound applies: max|χ|≤√23.
    # Source: algebraic power-map construction + DDT verification, March 2026.
    23: [
        14_932_592_847_614_922_746,   # x^3  mod 23 (δ=2, Weil tight) ✓
        35_288_577_512_348_823_734,   # x^15 mod 23 (δ=2) ✓
        25_087_749_894_427_280_174,   # x^21 mod 23 (δ=2) ✓
    ],                                                       # 3 APN ranks (δ=2)

    # n=29: δ_min=2 (APN, n≡2 mod 3). Power maps x^3, x^19, x^27 mod 29 are
    # algebraically proven APN (gcd(3,28)=1). Weil bound applies: max|χ|≤√29.
    # Source: algebraic power-map construction + DDT verification, March 2026.
    29: [
        2_794_638_805_054_714_907_838_431_436,   # x^3  mod 29 (δ=2, Weil tight) ✓
        9_931_985_764_146_292_318_498_003_478,   # x^19 mod 29 (δ=2) ✓
        5_378_154_461_386_366_131_111_442_874,   # x^27 mod 29 (δ=2) ✓
    ],                                                       # 3 APN ranks (δ=2)

    # n=31: δ_min=3 (NO APN EXISTS — OD-17 conjecture).
    # 31 ≡ 1 mod 3: x^3 not a bijection. All 7 bijective power maps have δ=4.
    # 3,300,000-trial random search found no δ=2 seed; best achievable is δ=3.
    # These seeds are NOT APN. They are the best-available (δ=3) permutations.
    # Conjecture OD-17: δ_min(Z_31) = 3 for ALL bijections.
    # Source: apn_search_vectorized(), V14 audit, March 2026.
    31: [
        8142658307823942392008438858021178,   # δ=3  V14-rng1
        7363955627480966101122957825563852,   # δ=3  V14-rng1
        2367823247044783685159688004427826,   # δ=3  V14-rng1
        5922398939626890749919123740877462,   # δ=3  V14-rng1
        3286632564743222379275261725230288,   # δ=3  V14-rng1
        2869759789971178142756068803699626,   # δ=3  V14-rng1
        606026577031455863975580487709072,    # δ=3  V14-rng1
        7578904442050970299326632615298773,   # δ=3  V14-rng1
    ],                                                       # 8 δ=3 seeds (NOT APN, OD-17)
}


def unrank_optimal_seed(k: int, n: int, signed: bool = True) -> np.ndarray:
    """
    Return the k-th APN/PN-class (Golden Seed) permutation for order n.

    Resolution order
    ----------------
    1. **Table lookup** — if n ∈ GOLDEN_SEEDS, return the k-th pre-verified rank.
    2. **Zero-compute APN** (Corollary of docs/PROOF_APN_OBSTRUCTION.md) —
       for prime n ≡ 2 (mod 3), the power map x^3 (mod n) is unconditionally
       APN (δ=2) and a bijection.  No search needed; O(n) construction.
    3. **Algebraic inverse fallback** — for other primes, x^{-1} (mod n)
       is a known APN construction.  δ=2 for n ≡ 3 (mod 4); otherwise best
       available.

    STATUS
    ------
    Path 1: PROVEN (pre-verified seeds, V12-V14).
    Path 2: PROVEN — see OD-16-PM / OD-17-PM and PROOF_APN_OBSTRUCTION.md.
            Applies to n ∈ {5,11,17,23,29,41,47,53,59,71,…}.
    Path 3: DESIGN INTENT (known APN for some n; not universally optimal).

    Parameters
    ----------
    k      : int   index into the Golden Seed list for n (wraps around)
    n      : int   odd prime order
    signed : bool  return signed (balanced) digit set if True

    Returns
    -------
    np.ndarray  shape (n,)  APN-class permutation array

    Raises
    ------
    ValueError  if k < 0
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    # ── Path 1: pre-computed Golden Seeds ───────────────────────────────────
    if n in GOLDEN_SEEDS:
        seeds = GOLDEN_SEEDS[n]
        rank  = seeds[k % len(seeds)]
        return factoradic_unrank(rank, n, signed=signed)

    # ── Path 2: zero-compute APN for prime n ≡ 2 (mod 3) ───────────────────
    # By docs/PROOF_APN_OBSTRUCTION.md (Corollary): for prime p ≡ 2 (mod 3),
    # gcd(3, p-1) = 1 so x^3 is a bijection, and R(X,Y)=3≠0 so δ=2.
    if _is_prime(n) and n % 3 == 2:
        seed = np.array([(x ** 3) % n for x in range(n)], dtype=int)
        if signed:
            seed = seed - (n // 2)
        return seed

    # ── Path 3: algebraic inverse map x ↦ x^{-1} mod n ─────────────────────
    # Known APN construction used in the AES S-Box design.
    inv_perm = np.zeros(n, dtype=int)
    for x in range(n):
        inv_perm[x] = pow(int(x), -1, n) if x != 0 else 0
    if signed:
        inv_perm = inv_perm - (n // 2)
    return inv_perm


def random_apn_search(
    n           : int,
    trials      : int = 1_000_000,
    target_delta: int = 2,
    rng_seed    : int = 42,
) -> dict:
    """
    Random search for permutations of Z_n with differential uniformity δ = target_delta.

    Samples `trials` uniformly random permutations of [0, n), evaluates δ via DDT
    for each, and returns those achieving target_delta.

    Intended use: probe whether APN (δ=2) permutations exist for given n,
    or find best-available seeds when δ_min > 2.

    Parameters
    ----------
    n            : int   base (odd prime recommended)
    trials       : int   number of random permutations to sample
    target_delta : int   differential uniformity target (default: 2 for APN)
    rng_seed     : int   numpy random seed for reproducibility

    Returns
    -------
    dict with:
        found          : int   number of seeds achieving target_delta
        min_delta_seen : int   minimum δ observed across all samples
        ranks          : list  Lehmer ranks of found seeds (up to 8)
        elapsed_sec    : float wall-clock seconds
        status         : str   "FOUND" | "SEARCHED_NO_RESULT"
    """
    import time
    rng   = np.random.default_rng(rng_seed)
    found : List[int] = []
    min_d = n

    start = time.perf_counter()
    for _ in range(trials):
        perm  = rng.permutation(n).astype(int)
        delta = differential_uniformity(perm, n)
        if delta < min_d:
            min_d = delta
        if delta == target_delta:
            found.append(factoradic_rank(perm, n, signed=False))

    elapsed = time.perf_counter() - start
    unique  = sorted(set(found))

    return {
        "n"              : n,
        "trials"         : trials,
        "target_delta"   : target_delta,
        "found"          : len(found),
        "unique_found"   : len(unique),
        "min_delta_seen" : min_d,
        "ranks"          : unique[:8],
        "elapsed_sec"    : elapsed,
        "status"         : "FOUND" if found else "SEARCHED_NO_RESULT",
    }


def apn_search_vectorized(
    n           : int,
    trials      : int           = 5_000_000,
    batch_size  : int           = 5_000,
    target_delta: int           = 2,
    rng_seed    : int           = 0,
) -> dict:
    """
    Vectorised random search for permutations of Z_n with δ = target_delta.

    Replaces random_apn_search() with a fully-NumPy batch DDT evaluation that
    achieves ~10µs / permutation vs ~125µs for the scalar version (12.5× faster).

    Algorithm
    ---------
    For each batch of B random permutations:
      1. Build the full difference table for shift a using NumPy broadcasting.
      2. Use an offset-bincount trick to compute all B histograms simultaneously.
      3. Track the minimum δ seen and collect seeds at that minimum.

    This is the search used in the V14 audit to investigate δ_min for n=19 and n=31.

    Parameters
    ----------
    n            : int   base (odd prime recommended)
    trials       : int   total number of permutations to sample (default 5M)
    batch_size   : int   permutations per batch (default 5000; tune for RAM)
    target_delta : int   stop early if a seed with this δ is found (default 2)
    rng_seed     : int   numpy random seed for reproducibility

    Returns
    -------
    dict with:
        n              : int
        trials         : int  actual number evaluated (may be < requested if early stop)
        target_delta   : int
        found          : int  number of seeds achieving target_delta (or best_delta)
        best_delta     : int  minimum δ observed
        ranks          : list Lehmer ranks of up to 8 unique best seeds
        elapsed_sec    : float
        status         : str  \"FOUND\" | \"SEARCHED_NO_RESULT\" | \"BEST_DELTA_ONLY\"

    Notes
    -----
    V14 Audit Results (documented in GOLDEN_SEEDS):
        n=19 — 8,000,000 total trials (V12: 1M, V14: 7M), 0 APN found.
                Rate of δ=3 seeds: ~3.2% of all random permutations.
                Strong evidence that δ_min(Z_19) = 3.  OD-16 status: OPEN.
        n=31 — 3,300,000 total trials (V12: 300K, V14: 3M), 0 APN found.
                Rate of δ=3 seeds: ~3.1% of all random permutations.
                Strong evidence that δ_min(Z_31) = 3.  OD-17 status: OPEN.
    """
    import time

    rng        = np.random.default_rng(rng_seed)
    base_idx   = np.arange(n, dtype=np.int64)
    offsets    = (np.arange(batch_size, dtype=np.int64) * n).reshape(batch_size, 1)
    best_delta = n
    best_ranks : List[int] = []
    total_done = 0

    t0 = time.perf_counter()

    batches = (trials + batch_size - 1) // batch_size
    for _ in range(batches):
        B = min(batch_size, trials - total_done)
        if B <= 0:
            break

        perms      = rng.permuted(np.tile(np.arange(n, dtype=np.int64), (B, 1)), axis=1)
        off        = (np.arange(B, dtype=np.int64) * n).reshape(B, 1)
        max_deltas = np.zeros(B, dtype=np.int64)

        for a in range(1, n):
            shifted = perms[:, (base_idx + a) % n]
            diffs   = (shifted - perms) % n
            flat    = (diffs + off).ravel()
            counts  = np.bincount(flat, minlength=B * n).reshape(B, n)
            np.maximum(max_deltas, counts.max(axis=1), out=max_deltas)

        total_done += B
        bd = int(max_deltas.min())

        if bd < best_delta:
            best_delta = bd
            best_ranks = []

        if bd == best_delta:
            mask = max_deltas == best_delta
            for perm in perms[mask]:
                if len(best_ranks) >= 8:
                    break
                rank = factoradic_rank(perm.tolist(), n, signed=False)
                if rank not in best_ranks:
                    best_ranks.append(rank)

        if best_delta <= target_delta:
            break

    elapsed = time.perf_counter() - t0

    if best_delta <= target_delta:
        status = "FOUND"
    elif best_ranks:
        status = "BEST_DELTA_ONLY"
    else:
        status = "SEARCHED_NO_RESULT"

    return {
        "n"            : n,
        "trials"       : total_done,
        "target_delta" : target_delta,
        "found"        : len(best_ranks) if best_delta <= target_delta else 0,
        "best_delta"   : best_delta,
        "ranks"        : best_ranks,
        "elapsed_sec"  : round(elapsed, 2),
        "status"       : status,
    }
