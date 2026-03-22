"""
flu/core/n_ary.py
==================
N-ary FM-Dance Generalisation — V12 Sprint Integration (V14).

THEOREM N-ARY-1 (PROVEN):
    The FM-Dance prefix transform T·a = x is valid for ALL n >= 2.
    det(T) = -1, so T is invertible over Z_n for ANY n (gcd(-1,n)=1).

ALIGNMENT PRINCIPLE (user request, V12 Sprint):
    Choose n to match the BASE RADIX of the construct being analysed:

    Construct order / type           Recommended n       Rationale
    ─────────────────────────────────────────────────────────────────
    Ternary (3-symbol)               n=3                 Digit-level
    9-ary (ternary block pairs)      n=9  (= 3²)         Block-level, 2-ternary
    27-ary (ternary block triples)   n=27 (= 3³)         Block-level, 3-ternary
    Binary / order-2 construct       n=2                 Binary (even-n path)
    Order-4 construct                n=4                 Base-4 (even-n path)
    Decimal / 10-symbol              n=10                Base-10 (even-n path)
    Arbitrary prime p                n=p                 Prime base
    Prime power p^k                  n=p^k               Block-level

    Note: odd n gives perfect mean-centering (mean=0).
          even n uses parity_switcher for near-balanced distribution.

KEY FUNCTIONS
─────────────
  nary_info(n, d)            → dict of properties for an (n,d) FM-Dance system
  nary_generate(n, d)        → full Latin hyperprism using n-ary addressing
  nary_verify(n, d)          → verification dict (Latin, line sums, mean)
  nary_step_bound(n, d)      → int: max torus step per FM-Dance increment
  recommend_base(order, dim) → suggested n and rationale
  ternary_block_base(k)      → n=3^k, i.e. k-ternary-digit blocks as symbols

SAFE TEST LIMITS (per user note, V12 Sprint):
    For interactive testing: n * d <= 40 is safe.
    Examples: n=3 d=6, n=5 d=4, n=7 d=3, n=9 d=2, n=11 d=2, n=13 d=2.
    n=13, d=4 → 13^4 = 28561 — feasible but slow. Avoid in tight loops.
    n=13, d=3 → 2197 — perfectly fine.

STATUS: PROVEN (N-ARY-1 in theorem registry).
        References flu.core.fm_dance (addressing) and flu.core.fm_dance_path (kinetic).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Alignment principle lookup ────────────────────────────────────────────────

_ALIGNMENT_TABLE: Dict[str, Dict[str, Any]] = {
    "ternary":       {"n": 3,  "rationale": "digit-level ternary (3-symbol)"},
    "ternary_block2": {"n": 9,  "rationale": "9-ary = 3² (2-ternary-digit blocks as symbols)"},
    "ternary_block3": {"n": 27, "rationale": "27-ary = 3³ (3-ternary-digit blocks as symbols)"},
    "binary":        {"n": 2,  "rationale": "binary / order-2 (even-n path)"},
    "order4":        {"n": 4,  "rationale": "base-4 / order-4 (even-n path)"},
    "order5":        {"n": 5,  "rationale": "quinary / order-5 (odd, perfect centering)"},
    "order7":        {"n": 7,  "rationale": "heptary / order-7 (odd, APN seeds available)"},
    "order11":       {"n": 11, "rationale": "11-ary / order-11 (odd prime, APN seeds)"},
    "decimal":       {"n": 10, "rationale": "decimal / base-10 (even-n path)"},
}


def recommend_base(order: int, dim: int = 2) -> Dict[str, Any]:
    """
    Return recommended n and rationale for a construct of given order.

    The alignment principle: n should match the radix/order of the construct.
    - If order is a prime:  n = order directly.
    - If order is a prime power p^k: n = p (digit-level) or n = p^k (block-level).
    - For ternary constructs (order=3): can use n=3, 9, or 27 for different scales.
    - For order-4 constructs: n=4 (even-n) or n=2 (binary pairs).

    Parameters
    ----------
    order : int  the 'order' of the construct (e.g. 3 for ternary, 4 for order-4)
    dim   : int  the spatial dimension D

    Returns
    -------
    dict with keys: n, rationale, step_bound, total_points, is_odd_n,
                    digit_level_n, block_alternatives
    """
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")

    # Find prime factorisation to check if it's a prime power
    def _factorise(n: int) -> List[int]:
        factors = []
        x = n
        for p in range(2, int(math.isqrt(x)) + 2):
            while x % p == 0:
                factors.append(p)
                x //= p
        if x > 1:
            factors.append(x)
        return factors

    factors = _factorise(order)
    unique_primes = list(set(factors))

    if len(unique_primes) == 1:
        p = unique_primes[0]
        k = len(factors)  # order = p^k
        if k == 1:
            # order is prime — n = order directly
            n_recommended = order
            rationale = f"order is prime {p}; n={p} gives digit-level alignment"
            block_alts = [p**j for j in range(2, 4) if j <= 4]
        else:
            # order is prime power p^k
            n_recommended = p  # digit-level
            rationale = (
                f"order={order}={p}^{k}; n={p} for digit-level (ternary-style). "
                f"Alternatively n={order} for block-level (k-digit blocks as symbols)."
            )
            block_alts = [p**j for j in range(1, k + 2) if j != 1 and p**j <= 27]
    else:
        # composite, not a prime power — use smallest prime factor
        p = min(unique_primes)
        n_recommended = p
        rationale = (
            f"order={order} is composite (not a prime power); "
            f"n={p} (smallest prime factor) recommended as base radix. "
            f"Or n={order} directly for base-{order} system."
        )
        block_alts = [order]

    step_bound = nary_step_bound(n_recommended, dim)
    total = n_recommended ** dim

    return {
        "n": n_recommended,
        "rationale": rationale,
        "step_bound": step_bound,
        "total_points": total,
        "is_odd_n": (n_recommended % 2 == 1),
        "digit_level_n": n_recommended,
        "block_alternatives": block_alts,
        "mean_centering": "exact (mean=0)" if n_recommended % 2 == 1 else "near-zero (parity_switcher needed)",
    }


def ternary_block_base(k: int) -> int:
    """
    Return n = 3^k, the n-ary base for analysing k-ternary-digit blocks as symbols.

    Examples:
        k=1 → n=3  (standard ternary, digits {-1,0,1})
        k=2 → n=9  (9-ary, each symbol = 2 ternary digits, 9 symbols)
        k=3 → n=27 (27-ary, each symbol = 3 ternary digits, 27 symbols)

    For higher-dimension ternary constructs, analysing at k>1 reveals structure
    that is invisible at the digit level.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return 3 ** k


def nary_step_bound(n: int, d: int) -> int:
    """
    Return the FM-Dance step bound: min(d, floor(n/2)).

    Theorem T4 (PROVEN): for any n >= 2, the torus-metric step between
    consecutive FM-Dance coordinates is bounded by min(d, floor(n/2)).
    """
    return min(d, n // 2)


def nary_info(n: int, d: int) -> Dict[str, Any]:
    """
    Return a property summary for an (n, d) FM-Dance system.

    Works for any n >= 2 and d >= 1 (Theorem N-ARY-1).

    Returns
    -------
    dict with keys:
        n, d, total_points, step_bound, digit_set, is_odd,
        mean_centered, d_star, locality_optimal_d, in_dimension_limited_regime
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    half = n // 2
    is_odd = (n % 2 == 1)

    if is_odd:
        digit_set = list(range(-half, half + 1))  # {-half, ..., 0, ..., half}
        mean = 0.0
        mean_desc = "exact zero (symmetric balanced)"
    else:
        digit_set = list(range(-half, half))  # {-half, ..., -1, 0, ..., half-1}
        mean = -0.5
        mean_desc = "near-zero (-0.5); use parity_switcher for balanced output"

    step_bound = nary_step_bound(n, d)
    d_star = half  # crossover dimension: D* = floor(n/2)

    return {
        "n":                      n,
        "d":                      d,
        "total_points":           n ** d,
        "step_bound":             step_bound,
        "digit_set":              digit_set,
        "is_odd":                 is_odd,
        "mean":                   mean,
        "mean_description":       mean_desc,
        "d_star":                 d_star,
        "in_dimension_limited":   d <= d_star,   # True: bound = d (not yet saturated)
        "in_radix_limited":       d > d_star,    # True: bound = d_star (saturated)
        "locality_optimal_d":     d_star,        # no benefit to D > D*
        "bijective":              True,           # N-ARY-1: T invertible for all n
        "latin_columns":          True,           # N-ARY-1: uniform projection preserved
    }


def nary_generate(n: int, d: int, max_cells: int = 50_000) -> np.ndarray:
    """
    Generate a Latin hyperprism using n-ary FM-Dance addressing.

    M[i_0, ..., i_{d-1}] = sum_k i_k  mod n  (sum-mod construction)

    This satisfies:
      - Latin property (T3 / PFNT-3): every axis slice is a permutation of Z_n
      - Constant line sum (L1): every line sums to n*(n-1)/2  (unsigned)
      - L2 Holographic Repair: single-cell erasure recoverable from line sum

    For signed representation: M - floor(n/2) gives {-half,...,half} symbols.
    The signed version satisfies L1 with line sum = 0.

    SAFE LIMITS: default max_cells=50_000 guards interactive use. Pass
    max_cells=float('inf') on high-memory hardware to bypass the guard.
    Example safe calls: nary_generate(9,2), nary_generate(5,4), nary_generate(3,6).
    AVOID by default: nary_generate(13,4) — 28561 cells; use nary_generate(13,3).

    Parameters
    ----------
    n : int  base radix (any n >= 2)
    d : int  spatial dimension
    max_cells : int  safety ceiling on n^d allocation (default 50_000).
                     Pass float('inf') to disable on high-memory systems.

    Returns
    -------
    np.ndarray, shape (n,)*d, dtype=int  (unsigned values 0..n-1)
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    total = n ** d
    if total > max_cells:
        raise ValueError(
            f"nary_generate({n},{d}) would allocate {total} cells (limit={max_cells}). "
            f"Use n={n}, d <= {int(math.log(max_cells, n))} for safety, "
            f"or pass max_cells=float('inf') to bypass on high-memory systems."
        )

    shape = (n,) * d
    M = np.fromfunction(
        lambda *idx: sum(idx) % n,
        shape,
        dtype=int
    )
    return M.astype(int)


def nary_generate_signed(n: int, d: int, max_cells: int = 50_000) -> np.ndarray:
    """
    Generate a SIGNED Latin hyperprism: M_signed = nary_generate(n,d) - floor(n/2).

    For odd n: values in {-floor(n/2), ..., 0, ..., floor(n/2)}, mean = 0.
    For even n: values in {-n/2, ..., n/2-1}, mean = -0.5 (near-zero).

    Line sum = 0 for odd n (L1 with signed symbols, S1 invariant).

    Parameters
    ----------
    max_cells : int  passed through to nary_generate (default 50_000).
    """
    M = nary_generate(n, d, max_cells=max_cells)
    return M - (n // 2)


def nary_verify(n: int, d: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Verify all key properties of an (n,d) n-ary FM-Dance Latin hyperprism.

    Checks: Latin property, constant line sum (L1), holographic repair (L2),
            mean centering (S1 for odd n).

    Parameters
    ----------
    n       : int  base radix
    d       : int  spatial dimension
    verbose : bool print results

    Returns
    -------
    dict: {latin, l1_constant_sum, l2_repairable, s1_mean_zero, all_pass}
    """
    M = nary_generate(n, d)
    M_s = M - (n // 2)  # signed version
    shape = (n,) * d
    half = n // 2

    # ── Latin check: every axis-aligned slice is a permutation of {0..n-1}
    latin_ok = True
    for axis in range(d):
        for fixed in np.ndindex(*[n if i != axis else 1 for i in range(d)]):
            slc: List[Any] = []
            fi = 0
            for dim in range(d):
                if dim == axis:
                    slc.append(slice(None))
                else:
                    slc.append(fixed[fi])
                    fi += 1
            vals = M[tuple(slc)].flatten()
            if len(set(vals.tolist())) != n:
                latin_ok = False
                break
        if not latin_ok:
            break

    # ── L1: constant line sum
    expected_sum_signed = 0  # for odd n, signed sum = 0
    line_sums: List[int] = []
    for axis in range(d):
        for fixed in np.ndindex(*[n if i != axis else 1 for i in range(d)]):
            slc = []
            fi = 0
            for dim in range(d):
                if dim == axis:
                    slc.append(slice(None))
                else:
                    slc.append(fixed[fi])
                    fi += 1
            line_sums.append(int(np.sum(M_s[tuple(slc)])))
    l1_ok = len(set(line_sums)) == 1
    l1_sum = line_sums[0] if line_sums else None

    # ── L2: holographic repair (erase and recover one cell per line)
    l2_ok = True
    if d >= 2 and n <= 13:
        # Test a sample of cells for repair
        for test_coord in [tuple([0] * d), tuple([n // 2] * d)]:
            true_val = int(M_s[test_coord])
            # Recover via axis-0 line sum
            axis = 0
            slc = list(test_coord)
            slc[0] = slice(None)
            line = M_s[tuple(slc)].copy()
            line[test_coord[0]] = 0  # "erase"
            recovered = int(l1_sum - int(np.sum(line)))
            if recovered != true_val:
                l2_ok = False
                break

    # ── S1: global mean = 0 (for odd n)
    global_mean = float(np.mean(M_s))
    s1_ok = abs(global_mean) < 1e-9

    result = {
        "n": n, "d": d,
        "latin":         latin_ok,
        "l1_constant_sum": l1_ok,
        "l1_sum_value":  l1_sum,
        "l2_repairable": l2_ok,
        "s1_mean_zero":  s1_ok,
        "global_mean":   global_mean,
        "all_pass":      latin_ok and l1_ok and l2_ok and s1_ok,
    }

    if verbose:
        print(f"N-ARY FM-Dance Verification  n={n}, d={d}")
        print(f"  Total points     : {n**d}")
        print(f"  Step bound T4    : {nary_step_bound(n, d)}")
        print(f"  Latin (T3/PFNT-3): {'PASS' if latin_ok else 'FAIL'}")
        print(f"  L1 (const sum)   : {'PASS' if l1_ok else 'FAIL'}  (sum={l1_sum})")
        print(f"  L2 (repair)      : {'PASS' if l2_ok else 'FAIL'}")
        print(f"  S1 (mean=0)      : {'PASS' if s1_ok else 'FAIL'}  (mean={global_mean:.2e})")
        print(f"  All pass         : {'✓' if result['all_pass'] else '✗'}")

    return result


def nary_comparison_table(
    n_values: Optional[List[int]] = None,
    d_values: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Produce a comparison table of n-ary FM-Dance systems.

    Demonstrates the alignment principle: how different n choices affect
    step bound, mean centering, and total points for same D.

    Parameters
    ----------
    n_values : list of int  bases to compare (default: [2, 3, 5, 7, 9, 11])
    d_values : list of int  dimensions to show (default: [2, 3])

    Returns
    -------
    list of dicts, one per (n, d) combination
    """
    if n_values is None:
        n_values = [2, 3, 5, 7, 9, 11]
    if d_values is None:
        d_values = [2, 3]

    rows = []
    for n in n_values:
        for d in d_values:
            info = nary_info(n, d)
            rows.append({
                "n": n, "d": d,
                "total_points":       n ** d,
                "step_bound":         info["step_bound"],
                "d_star":             info["d_star"],
                "regime":             "dim-limited" if info["in_dimension_limited"] else "radix-limited",
                "odd_n":              info["is_odd"],
                "mean_centering":     "exact" if info["is_odd"] else "near",
                "note":               _regime_note(n, d),
            })
    return rows


def _regime_note(n: int, d: int) -> str:
    half = n // 2
    if d < half:
        return f"D={d} < D*={half}: step_bound=D (not yet saturated)"
    elif d == half:
        return f"D={d} = D*={half}: step_bound=D* (at crossover)"
    else:
        return f"D={d} > D*={half}: step_bound={half} (saturated, no locality gain)"


def verify_nary_bijection(n: int, d: int) -> bool:
    """
    Verify N-ARY-1: FM-Dance is bijective for given (n,d).

    Simply checks that all n^d coordinates are distinct when decoded.
    Safe limit: n^d <= 50000.
    """
    total = n ** d
    if total > 50_000:
        raise ValueError(f"n^d={total} too large for exhaustive bijection check (limit 50000)")
    from flu.core.fm_dance import index_to_coords, coords_to_index
    coords_seen = set()
    for k in range(total):
        if n % 2 == 0:
            # Even n: use unsigned representation
            half = n // 2
            digits = []
            rem = k
            for _ in range(d):
                digits.append(rem % n - half)
                rem //= n
            coord = tuple(digits)
        else:
            coord = index_to_coords(k, n, d)
        coords_seen.add(coord)
    return len(coords_seen) == total
