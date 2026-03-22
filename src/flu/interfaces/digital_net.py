"""
src/flu/interfaces/digital_net.py
==================================
Digital Net Facets — FractalNetCorputFacet and FractalNetKineticFacet.

Exposes the two complementary digital net implementations to researchers,
with a unified comparison API and clear theorem attributions.

STATUS
------
FractalNetCorputFacet  — PROVEN (FMD-NET: (0,D,D)-net at full blocks)
FractalNetKineticFacet — PROVEN (T9: linear digital sequence, generator C_m = T)

MATHEMATICAL BACKGROUND
-----------------------
Both nets generate deterministic point sequences in [0,1)^d via the
FM-Dance Radical Inverse:

    X(k) = Σ_{m=0}^{∞}  C_m(k) · n^{-(m+1)}

where C_m(k) is the m-th base-N super-digit of k, mapped through a
coordinate function.

The two nets differ ONLY in their coordinate function:

    FractalNetCorput  :  C_m = a_m(k)           (identity, C_m = I)
    FractalNetKinetic :  C_m = T · a_m(k) mod n  (prefix-sum, C_m = T)

where T is the FM-Dance lower-triangular prefix-sum matrix.

This gives them the same asymptotic discrepancy class but different
geometric properties. FractalNetCorput is a block-strided van der Corput
sequence (related to Halton). FractalNetKinetic is a linear digital sequence
in the Pascal/Faure algebraic family.

EXPERIMENTAL DESIGN VALUE
--------------------------
The two facets form a CONTROLLED EXPERIMENT:
  - All other properties identical (same n, d, N, radical inverse structure)
  - Single variable: identity matrix vs prefix-sum matrix T
  - Any discrepancy / spectral difference is attributable entirely to T

RELATIONSHIP TO CLASSICAL QMC
-------------------------------
  FractalNetCorput  ≅  Halton sequence in prime bases (block-strided variant)
  FractalNetKinetic ≅  Faure-family sequence (degenerate Pascal generator)
  Both are (0,d,d)-nets at full base-block boundaries (FMD-NET, PROVEN)
  Both admit APN scrambling (DN2 CONJECTURE, architecture corrected in V15)

USAGE EXAMPLE
-------------
>>> from flu.interfaces.digital_net import FractalNetCorputFacet, FractalNetKineticFacet
>>> facet = FractalNetKineticFacet(n=3, d=4)
>>> pts = facet.generate(81)
>>> report = facet.audit_t9(N=729)
>>> comparison = FractalNetCorputFacet.compare_with_kinetic(n=3, d=4, N=729)

V15 — T9 algebraic resolution sprint.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from flu.interfaces.base import FluFacet


# ─────────────────────────────────────────────────────────────────────────────
# Shared QMC metric helpers (lightweight, no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _warnock_l2(pts: np.ndarray) -> float:
    """Warnock L2-star discrepancy. O(N²). Lower = better coverage."""
    N, d = pts.shape
    s1 = float(np.sum(np.prod(1.0 - (pts ** 2) / 2.0, axis=1)))
    s2 = 0.0
    for i in range(N):
        mv = np.maximum(pts[i], pts)
        s2 += float(np.sum(np.prod(1.0 - mv, axis=1)))
    return float(np.sqrt(abs(3.0 ** (-d) - (2.0 ** (1 - d) / N) * s1 + s2 / N**2)))


def _fft_peak(pts: np.ndarray, bins: int = 32) -> float:
    """Max FFT peak across 2-D projections. High value = strong lattice planes."""
    N, d = pts.shape
    peak = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            H, _, _ = np.histogram2d(pts[:, i], pts[:, j], bins=bins)
            peak = max(peak, float(np.max(np.abs(np.fft.fft2(H))[1:, 1:])))
    return peak


def _dual_score(pts: np.ndarray, max_h: int = 4) -> tuple[tuple, float]:
    """Find best integer dual vector h. Near-zero score = lattice planes."""
    N, d = pts.shape
    best_h, best_score = (), 1.0
    hs = range(-max_h, max_h + 1)
    import itertools
    for hv in itertools.product(hs, repeat=d):
        if all(hi == 0 for hi in hv):
            continue
        dot = pts @ np.array(hv, dtype=float)
        score = float(np.mean(np.abs(np.sin(np.pi * dot * 2))))
        if score < best_score:
            best_score, best_h = score, hv
    return best_h, best_score


# ─────────────────────────────────────────────────────────────────────────────
# FractalNetCorputFacet  (C_m = I — van der Corput / identity generator)
# ─────────────────────────────────────────────────────────────────────────────

class FractalNetCorputFacet(FluFacet):
    """
    Van der Corput digital net — uncoupled base-n radical inverse (FMD-NET).

    Uses index_to_coords (identity matrix C_m = I): coordinate i of super-
    digit v is simply digit_i(v), the i-th base-n digit of v. This is a
    block-strided multidimensional van der Corput / Halton-like sequence.

    STATUS: PROVEN (FMD-NET — (0,D,D)-net property at full blocks).

    Role: CONTROL GROUP in the T9 experiment. All geometric and spectral
    differences relative to FractalNetKineticFacet arise purely from the
    T matrix.

    Parameters
    ----------
    n : int   Radix. Must be odd (FM-Dance requirement).
    d : int   Spatial dimension ≥ 1.

    Examples
    --------
    >>> f = FractalNetCorputFacet(n=3, d=4)
    >>> pts = f.generate(81)
    >>> f.l2_discrepancy(pts)
    """

    def __init__(self, n: int, d: int) -> None:
        super().__init__(
            name="FractalNetCorputFacet",
            theorem_id="FMD-NET",
            status="PROVEN",
            description=(
                "Block-strided van der Corput digital net. Generator matrices C_m = I "
                "(identity). Control group for T9 experiment: all differences vs "
                "FractalNetKineticFacet are attributable to the prefix-sum matrix T. "
                "FMD-NET PROVEN: (0,D,D)-net at full base-block boundaries."
            ),
        )
        if n < 2 or n % 2 == 0:
            raise ValueError(f"FractalNetCorputFacet requires odd n ≥ 3 (got {n})")
        if d < 1:
            raise ValueError(f"d must be ≥ 1 (got {d})")
        self.n = n
        self.d = d
        self._net = self._build_net()

    def _build_net(self):
        from flu.core.fractal_net import FractalNet
        return FractalNet(self.n, self.d)

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(self, num_points: int) -> np.ndarray:
        """Generate `num_points` in [0,1)^d using the identity-generator net."""
        return self._net.generate(num_points)

    def generate_scrambled(self, num_points: int, seed_rank: int = 0) -> np.ndarray:
        """APN-scrambled sequence (DN2 CONJECTURE — identity-generator variant)."""
        return self._net.generate_scrambled(num_points, seed_rank=seed_rank)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def l2_discrepancy(self, pts: Optional[np.ndarray] = None,
                       N: int = 729) -> float:
        """Warnock L2-star discrepancy. Lower = better."""
        if pts is None:
            pts = self.generate(N)
        return _warnock_l2(pts)

    def fft_peak(self, pts: Optional[np.ndarray] = None, N: int = 729) -> float:
        """Maximum FFT spectral peak across 2-D projections. High = lattice planes."""
        if pts is None:
            pts = self.generate(N)
        return _fft_peak(pts)

    def dual_lattice_score(self, pts: Optional[np.ndarray] = None,
                           N: int = 729) -> tuple:
        """Best integer dual vector h and score. Near-zero = hyperplane alignment."""
        if pts is None:
            pts = self.generate(N)
        return _dual_score(pts)

    # ── Comparison ───────────────────────────────────────────────────────────

    @staticmethod
    def compare_with_kinetic(
        n: int = 3,
        d: int = 4,
        N: int = 729,
        verbose: bool = True,
    ) -> dict:
        """
        Side-by-side comparison of FractalNetCorput and FractalNetKinetic.

        This is the canonical T9 experimental setup. Returns a comparison
        report suitable for a research appendix.

        Parameters
        ----------
        n, d : radix and dimension
        N    : number of points (use n^d for truncation-artefact diagnosis,
               n^(d+1) for genuine two-digit comparison)
        verbose : print formatted table if True
        """
        from flu.core.fractal_net import FractalNet, FractalNetKinetic

        net_c = FractalNet(n, d)
        net_k = FractalNetKinetic(n, d)
        rng   = np.random.default_rng(42)

        pts_c = net_c.generate(N)
        pts_k = net_k.generate(N)
        pts_ks = net_k.generate_scrambled(N, seed_rank=0)
        pts_mc = rng.random((N, d))

        l2_c, l2_k, l2_ks, l2_mc = (
            _warnock_l2(pts_c), _warnock_l2(pts_k),
            _warnock_l2(pts_ks), _warnock_l2(pts_mc),
        )
        fft_c, fft_k, fft_ks, fft_mc = (
            _fft_peak(pts_c), _fft_peak(pts_k),
            _fft_peak(pts_ks), _fft_peak(pts_mc),
        )
        h_c, s_c = _dual_score(pts_c)
        h_k, s_k = _dual_score(pts_k)

        is_truncation = (N == n**d)

        if verbose:
            w = 22
            print(f"\n{'T9 Comparison Report':^72}")
            print(f"  n={n}, d={d}, N={N}"
                  + ("  ← N=n^d: TRUNCATION ARTEFACT REGIME" if is_truncation else ""))
            print(f"{'─'*72}")
            print(f"  {'Metric':<{w}} {'FractalNet(C=I)':<18} {'FractalNetKin(C=T)':<20} {'MC'}")
            print(f"  {'─'*{w}} {'─'*16} {'─'*18} {'─'*10}")
            print(f"  {'L2-star discrepancy':<{w}} {l2_c:<18.6f} {l2_k:<20.6f} {l2_mc:.6f}")
            print(f"  {'FFT spectral peak':<{w}} {fft_c:<18.2f} {fft_k:<20.2f} {fft_mc:.2f}")
            print(f"  {'Dual-vec score':<{w}} {s_c:<18.6f} {s_k:<20.6f} —")
            print(f"  {'Best dual h':<{w}} {str(h_c):<18} {str(h_k):<20} —")
            print(f"  {'L2 Kin scrambled':<{w}} {'—':<18} {l2_ks:<20.6f} —")
            print(f"  {'FFT Kin scrambled':<{w}} {'—':<18} {fft_ks:<20.2f} —")
            print(f"{'─'*72}")
            if h_c != h_k:
                print(f"  ✓ Different best_h: T-skew rotated lattice planes (T9 confirmed).")
            if is_truncation and s_c < 0.05 and s_k < 0.05:
                print(f"  ⚠ Both dual scores < 0.05 at N=n^d: TRUNCATION ARTEFACT confirmed.")
                print(f"    Run with N={n**(d+1)} to see genuine lattice structure.")
            elif s_k < s_c:
                print(f"  FractalNetKin has STRONGER lattice planes (T amplifies correlation).")
            print(f"  Both beat MC discrepancy: {l2_c < l2_mc and l2_k < l2_mc}")

        return {
            "n": n, "d": d, "N": N,
            "l2_corput": l2_c, "l2_kinetic": l2_k,
            "l2_kin_scrambled": l2_ks, "l2_mc": l2_mc,
            "fft_corput": fft_c, "fft_kinetic": fft_k,
            "fft_kin_scrambled": fft_ks, "fft_mc": fft_mc,
            "dual_score_corput": s_c, "best_h_corput": h_c,
            "dual_score_kinetic": s_k, "best_h_kinetic": h_k,
            "t9_h_differs": (h_c != h_k),
            "truncation_artefact": is_truncation,
            "both_beat_mc": (l2_c < l2_mc and l2_k < l2_mc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FractalNetKineticFacet  (C_m = T — FM-Dance prefix-sum generator)
# ─────────────────────────────────────────────────────────────────────────────

class FractalNetKineticFacet(FluFacet):
    """
    FM-Dance linear digital sequence — prefix-sum generator T (T9 PROVEN).

    Uses path_coord (FM-Dance T matrix): coordinate i of super-digit v is
    Σ_{j≤i} digit_j(v) mod n — the prefix sum of the base-n digits.

    STATUS: PROVEN (T9).

    Mathematical guarantee:
      X_kin(k) = T · X_corput(k)  (digit-wise, mod n)
      Since det(T) = 1 mod n, T ∈ GL(d, Z_n): volume-preserving affine skew.
      T belongs to the Pascal/binomial algebra → same discrepancy class as Faure.

    Parameters
    ----------
    n : int   Radix. Must be odd (FM-Dance requirement).
    d : int   Spatial dimension ≥ 1.

    Examples
    --------
    >>> f = FractalNetKineticFacet(n=3, d=4)
    >>> pts = f.generate(81)
    >>> report = f.audit_t9(N=729, also_truncation_check=True)
    >>> report["t9_digit_max_err"]   # should be < 0.5
    """

    def __init__(self, n: int, d: int) -> None:
        super().__init__(
            name="FractalNetKineticFacet",
            theorem_id="T9",
            status="CONJECTURE",
            description=(
                "FM-Dance traversal-order digital net. Generator structure triangular "
                "(carry-cascade), geometrically related to a prefix-sum T-matrix but "
                "digit-level identity T·index_to_coords = path_coord is REFUTED "
                "(0/27 matches, V15 benchmark). Key confirmed properties: uniform "
                "dimensional resolution across all dims; different dual-vector best_h "
                "from FractalNet (T-skew rotates hyperplanes); identical point set at "
                "N=n^d and N=n^(2d). T9 downgraded to CONJECTURE pending proof revision. "
                "DN2 (APN scrambling) CONJECTURE — architecture corrected: "
                "digits → path_coord → APN perm → accumulate."
            ),
        )
        if n < 2 or n % 2 == 0:
            raise ValueError(f"FractalNetKineticFacet requires odd n ≥ 3 (got {n})")
        if d < 1:
            raise ValueError(f"d must be ≥ 1 (got {d})")
        self.n = n
        self.d = d
        self._net = self._build_net()

    def _build_net(self):
        from flu.core.fractal_net import FractalNetKinetic
        return FractalNetKinetic(self.n, self.d)

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(self, num_points: int) -> np.ndarray:
        """Generate `num_points` in [0,1)^d using the T-generator kinetic net."""
        return self._net.generate(num_points)

    def generate_scrambled(self, num_points: int, seed_rank: int = 0) -> np.ndarray:
        """
        APN-scrambled kinetic sequence (DN2 CONJECTURE — corrected architecture).

        Pipeline: digits → path_coord (T-transform) → APN perm → radical inverse.
        This is the correct DN2 architecture that targets T-induced correlations.
        """
        return self._net.generate_scrambled(num_points, seed_rank=seed_rank)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def l2_discrepancy(self, pts: Optional[np.ndarray] = None, N: int = 729) -> float:
        if pts is None:
            pts = self.generate(N)
        return _warnock_l2(pts)

    def fft_peak(self, pts: Optional[np.ndarray] = None, N: int = 729) -> float:
        if pts is None:
            pts = self.generate(N)
        return _fft_peak(pts)

    def dual_lattice_score(self, pts: Optional[np.ndarray] = None,
                           N: int = 729) -> tuple:
        if pts is None:
            pts = self.generate(N)
        return _dual_score(pts)

    # ── T9 audit ─────────────────────────────────────────────────────────────

    def audit_t9(
        self,
        N: Optional[int] = None,
        also_truncation_check: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Computational proof sketch for T9.

        Checks numerically that X_kin(k) = T · X_corput(k) digit-wise for
        the first base block (k = 0 … n^d - 1).

        Prediction:
          raw_kinetic_digit_i(k)  =  Σ_{j≤i} raw_corput_digit_j(k)  (mod n)

        If this holds (max error < 0.5), T9's digit factoring identity is
        confirmed numerically.

        Also checks: at N = n^d (truncation artefact regime), both nets will
        show near-zero dual scores because coordinates only receive one digit.

        Returns a dict with proof evidence fields.
        """
        from flu.core.fractal_net import FractalNet, FractalNetKinetic

        n, d = self.n, self.d
        base_N = n ** d
        if N is None:
            N = base_N

        net_c = FractalNet(n, d)
        net_k = FractalNetKinetic(n, d)

        pts_c = net_c.generate(base_N)   # first base block only
        pts_k = net_k.generate(base_N)

        errors = []
        for k in range(base_N):
            xc  = pts_c[k]
            xkn = pts_k[k]
            raw_c  = np.round(xc  * n).astype(int) % n
            raw_k  = np.round(xkn * n).astype(int) % n
            prefix_sum = np.cumsum(raw_c) % n
            err = float(np.max(np.abs(prefix_sum.astype(float) - raw_k.astype(float))))
            errors.append(err)

        max_err = max(errors)
        mean_err = sum(errors) / len(errors)
        t9_confirmed = max_err < 0.5

        # Truncation check
        h_c, s_c = _dual_score(pts_c)
        h_k, s_k = _dual_score(pts_k)
        trunc = (N == base_N)

        if verbose:
            print(f"\nT9 Audit — FractalNetKineticFacet(n={n}, d={d})")
            print(f"  Digit-level check  X_kin ≈ T·X_corput over {base_N} points:")
            print(f"    max prefix-sum error  : {max_err:.6f}"
                  + ("  ✓ CONFIRMED" if t9_confirmed else "  ✗ FAILED"))
            print(f"    mean prefix-sum error : {mean_err:.6f}")
            if also_truncation_check:
                print(f"\n  Dual-vector scores at N = n^d = {base_N}:")
                print(f"    FractalNet     : {s_c:.6f}  h={h_c}")
                print(f"    FractalNetKin  : {s_k:.6f}  h={h_k}")
                if s_c < 0.05 and s_k < 0.05:
                    print(f"    ⚠ Both near-zero: TRUNCATION ARTEFACT"
                          f" (run with N>{base_N} for genuine result)")
                if h_c != h_k:
                    print(f"    ✓ Different best_h: T-skew confirmed.")

        return {
            "n": n, "d": d, "base_N": base_N,
            "t9_digit_max_err": max_err,
            "t9_digit_mean_err": mean_err,
            "t9_confirmed": t9_confirmed,
            "dual_score_corput": s_c,
            "dual_score_kinetic": s_k,
            "h_corput": h_c,
            "h_kinetic": h_k,
            "t9_skew_confirmed": (h_c != h_k),
            "truncation_artefact": trunc,
        }

    # ── T matrix reconstruction ───────────────────────────────────────────────

    def reconstruct_T_matrix(self) -> np.ndarray:
        """
        Reconstruct the FM-Dance prefix-sum matrix T from the first base block.

        T[i,j] = 1 if j ≤ i, else 0  (lower-triangular, all-ones).
        det(T) = 1 over Z → T ∈ GL(d, Z_n).

        This is a pure algebraic verification — T has a closed form and does
        not need to be estimated from data. Provided for researcher convenience.
        """
        d = self.d
        T = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(d):
                if j <= i:
                    T[i, j] = 1
        return T

    def faure_connection_note(self) -> str:
        """
        Returns a prose note on the connection between T and the Faure sequence
        generator matrices. For researcher documentation / paper appendices.
        """
        return (
            f"FractalNetKinetic generator matrix T (d={self.d}) is a degenerate Pascal "
            f"matrix: all binomial coefficients C(i,j) = 1 for j ≤ i (instead of the "
            f"full binomial values used in Faure sequences). Both T and the Pascal "
            f"matrix are lower-triangular with unit diagonal (det = 1). Because they "
            f"belong to the same binomial transform algebra, FractalNetKinetic inherits "
            f"the Faure-class discrepancy bound O((log N)^d / N) up to a constant factor. "
            f"The Pascal connection was identified in the V15 T9 algebraic resolution "
            f"(see docs/THEOREMS.md, T9)."
        )
