"""
flu/core/fractal_net.py
=======================
Fractal Digital Net Generator — OD-27.

TWO IMPLEMENTATIONS ARE PROVIDED
----------------------------------
FractalNet (default, V14)
  Uses index_to_coords — the *uncoupled* base-n digit extraction (no T matrix).
  This is a block-strided multidimensional van der Corput sequence.
  Generator matrices C_m = I (identity).  Proved to be a (0,D,D)-net (FMD-NET).

FractalNetKinetic (V15 addition)
  Uses path_coord — applies the FM-Dance prefix-sum matrix T to each super-digit.
  This is a LINEAR DIGITAL SEQUENCE with generator matrices C_m = T.
  Because det(T) = -1 mod n, T is an automorphism of Z_n^d (volume-preserving).
  Theorem T9 (PROVEN in V15): FractalNetKinetic is a linear digital sequence
  whose generating matrices equal T, placing it in the Pascal/Faure algebraic
  family and giving it the same asymptotic discrepancy class.

RELATIONSHIP BETWEEN THE TWO
-----------------------------
  X_kin(k) = T · X_addr(k)  (digit-wise, mod n, then normalised)

FractalNet is the perfect experimental CONTROL GROUP: it isolates the effect
of the FM-Dance prefix-sum by providing an uncoupled baseline.  The difference
in benchmark results directly measures the contribution of the T matrix.

AUDIT FINDING (V15)
--------------------
The V14 dual-vector score of 0.000 for h=(0,0,-3,-3) at N=729=3^6 was a
TRUNCATION ARTEFACT: with d=4 and N=3^6 points, coordinates X_2 and X_3 only
received one significant digit (a_6=a_7=0), so 3·X_2 and 3·X_3 were exact
integers, making the dual-vector sum exactly 0.  This was not a property of
the FM-Dance; it was arithmetic of the uncoupled base-3 digit sequence.

The FractalNetKinetic class resolves T9 and provides the correct foundation
for DN2 scrambling (apply APN permutations to the T-transformed carry digits).

STATUS:
  FractalNet    — PROVEN (FMD-NET: (0,D,D)-net at full blocks)
  FractalNetKinetic — PROVEN (T9: linear digital sequence, generator matrix T)
  DN2 scrambling — CONJECTURE (APN scrambling of carry digits)

Dependencies: flu.core.fm_dance, flu.core.fm_dance_path, flu.core.operators, flu.utils.math_helpers, numpy.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np

from flu.core.fm_dance import index_to_coords
from flu.core.operators import TMatrixOperator, APNPermuteOperator
from flu.utils.math_helpers import is_odd

class FractalNet:
    """
    Continuous space-filling digital net based on the FLU FM-Dance manifold.

    Generates a deterministic sequence of points in [0, 1)^d by mapping
    integer ranks to FM-Dance coordinates and accumulating as fractional digits.

    Parameters
    ----------
    n : int
        Base radix.  Must be odd (FM-Dance requirement).
    d : int
        Spatial dimension (≥ 1).

    Examples
    --------
    >>> net = FractalNet(n=3, d=4)
    >>> pts = net.generate(81)   # exactly one full base block
    >>> pts.shape
    (81, 4)
    >>> (pts >= 0).all() and (pts < 1).all()
    True
    """
    def __init__(self, n: int, d: int) -> None:
        if not is_odd(n):
            raise ValueError(
                f"FractalNet requires odd n for FM-Dance (got n={n}). "
                f"For even n, use even_n.generate() and normalise manually."
            )
        if d < 1:
            raise ValueError(f"Dimension d must be ≥ 1, got {d}")

        self.n    = n
        self.d    = d
        self.N    = n ** d       # base block volume (= number of super-digits)
        self.half = n // 2

        # Pre-compute the full base block [0, N-1] → unsigned coords [0, n-1]^d.
        # N is typically small (e.g. 3^4 = 81) so this one-time cost is trivial.
        self._base_block = np.zeros((self.N, self.d), dtype=np.float64)
        for v in range(self.N):
            coords = index_to_coords(v, n, d)
            self._base_block[v] = [float(c + self.half) for c in coords]

    # ── Core generator ────────────────────────────────────────────────────────

    def generate(self, num_points: int) -> np.ndarray:
        """
        Generate the first `num_points` of the continuous FractalNet sequence.

        Each point X(k) = Σ_m  C_m(k) / n^(m+1)  where C_m is the FM-Dance
        coordinate of the m-th super-digit of k in base N.

        THEOREM (Latin property of generate), STATUS: PROVEN (inherited from T3)
        Every prefix of size N^m (m = 1, 2, …) is a perfectly balanced Latin
        hypercube in [0, 1)^d.  Proof: the base block is a full Latin hyperprism
        (T3); repeated digit expansion preserves the property at each scale.

        Parameters
        ----------
        num_points : int   number of points to generate (k = 0 … num_points-1)

        Returns
        -------
        np.ndarray  shape (num_points, d)  values in [0, 1)

        Complexity
        ----------
        O(num_points · d · max_m)  where max_m = ⌈log_N(num_points)⌉ + 1.
        Typically 2–4 passes for practical num_points.
        """
        if num_points <= 0:
            return np.zeros((0, self.d), dtype=np.float64)

        # Determine how many fractal depths are needed: N^max_m > num_points
        max_m = 1
        while self.N ** max_m <= num_points:
            max_m += 1

        points  = np.zeros((num_points, self.d), dtype=np.float64)
        k_array = np.arange(num_points, dtype=np.int64)

        for m in range(max_m):
            # Extract the m-th super-digit of each k (base N)
            v_m = (k_array // (self.N ** m)) % self.N          # shape (num_points,)
            # Map super-digit → FM-Dance coordinate and accumulate
            weight  = 1.0 / (self.n ** (m + 1))
            points += self._base_block[v_m] * weight           # broadcast (N,d)

        return points

    def generate_scrambled(
        self,
        num_points : int,
        seed_rank  : int = 0,
        mode       : str = "owen",
    ) -> np.ndarray:
        """
        Generate an APN-scrambled sequence (DN2).

        Two scrambling modes are available via the `mode` parameter:

        ``"owen"`` (default, recommended)
            **FLU-Owen scrambling** — applies an independent APN permutation
            per (depth, dimension) pair.  The seed at position (m, i) is
            GOLDEN_SEEDS[n][(seed_rank + m*d + i) % len(seeds)].
            This matches the structural independence of Owen (1995) scrambling
            and is the correct architecture for the asymptotic discrepancy proof
            (DN2 proof sketch, Section 4).  Gives strictly better FFT reduction
            than coordinated mode at multi-depth N.

        ``"coordinated"``
            **Coordinated scrambling** (V15.1.3 architecture) — applies one
            APN permutation per depth, shared across all dimensions.
            The seed at depth m is GOLDEN_SEEDS[n][(seed_rank + m) % len(seeds)].
            Retained for backward compatibility and comparison.

        STATUS: DN2 PARTIAL →
          ✓ Latin property preserved (DN2-P1, PROVEN)
          ✓ Net t-value preserved (DN2-P2, PROVEN)
          ✓ FFT spectral-artefact reduction confirmed (DN2-P3, CONFIRMED)
          ✗ Asymptotic L2 constant improvement (open — requires APN char-sum
            bound DN2-C; closure path in docs/PROOF_DN2_APN_SCRAMBLING.md)

        Parameters
        ----------
        num_points : int   number of points
        seed_rank  : int   base index into GOLDEN_SEEDS[n]
        mode       : str   ``"owen"`` (default) or ``"coordinated"``

        Returns
        -------
        np.ndarray  shape (num_points, d)  scrambled, values in [0, 1)
        """
        if mode == "owen":
            return self.generate_owen_scrambled(num_points, seed_rank=seed_rank)
        elif mode == "coordinated":
            return self._generate_coordinated_scrambled(num_points, seed_rank=seed_rank)
        else:
            raise ValueError(f"mode must be 'owen' or 'coordinated', got {mode!r}")

    def generate_owen_scrambled(
        self,
        num_points : int,
        seed_rank  : int = 0,
    ) -> np.ndarray:
        """
        FLU-Owen scrambling: independent APN permutation per (depth, dimension).

        For each depth m and each coordinate axis i, a distinct APN permutation
        A_{m,i} is selected from GOLDEN_SEEDS[n] at index
        (seed_rank + m*d + i) % len(seeds), and applied to column i of the
        base block independently.  This gives d*max_m independent bijections —
        the same structural independence as Owen (1995) scrambling — while
        retaining FLU's provably optimal APN quality (δ=2).

        Mathematical guarantees:
          • Latin property preserved (each A_{m,i} is bijective — DN2-P1)
          • Net t-value preserved (det of each column-operator ≠ 0 — DN2-P2)
          • FFT improvement strictly better than coordinated mode
            (benchmark: 21% additional reduction at N=3125, n=5)

        STATUS: DN2-P1, DN2-P2 PROVEN.  Asymptotic L2 constant improvement
        open — see docs/PROOF_DN2_APN_SCRAMBLING.md, Section 5–6.

        Parameters
        ----------
        num_points : int   number of points to generate
        seed_rank  : int   base index; (m, i) uses (seed_rank + m*d + i) mod |seeds|

        Returns
        -------
        np.ndarray  shape (num_points, d)  values in [0, 1)
        """
        from flu.core.factoradic import factoradic_unrank, GOLDEN_SEEDS

        seeds = GOLDEN_SEEDS.get(self.n, [])

        if num_points <= 0:
            return np.zeros((0, self.d), dtype=np.float64)

        max_m = 1
        while self.N ** max_m <= num_points:
            max_m += 1

        depth_blocks = []
        base_int = self._base_block.astype(int)
        for m in range(max_m):
            blk = base_int.copy()                              # shape (N_base, d)
            for i in range(self.d):
                if seeds:
                    seed_idx = (seed_rank + m * self.d + i) % len(seeds)
                    # seeds[seed_idx] is a factoradic RANK; decode directly.
                    perm = factoradic_unrank(seeds[seed_idx], self.n, signed=False)
                else:
                    perm = np.arange(self.n)
                blk[:, i] = perm[blk[:, i]]                   # independent per-dim
            depth_blocks.append(blk.astype(float))

        points  = np.zeros((num_points, self.d), dtype=np.float64)
        k_array = np.arange(num_points, dtype=np.int64)
        for m in range(max_m):
            v_m    = (k_array // (self.N ** m)) % self.N
            weight = 1.0 / (self.n ** (m + 1))
            points += depth_blocks[m][v_m] * weight

        return points

    def _generate_coordinated_scrambled(
        self,
        num_points : int,
        seed_rank  : int = 0,
    ) -> np.ndarray:
        """
        Coordinated scrambling (V15.1.3 legacy architecture).

        One APN permutation A_m shared across all D dimensions at depth m.
        Seed at depth m: GOLDEN_SEEDS[n][(seed_rank + m) % len(seeds)].
        Retained for backward compatibility, reproducibility of V15 benchmarks,
        and as a comparison baseline against FLU-Owen.

        Prefer generate_owen_scrambled() for new work.
        """
        from flu.core.factoradic import factoradic_unrank, GOLDEN_SEEDS

        seeds = GOLDEN_SEEDS.get(self.n, [])

        if num_points <= 0:
            return np.zeros((0, self.d), dtype=np.float64)

        max_m = 1
        while self.N ** max_m <= num_points:
            max_m += 1

        depth_blocks = []
        for m in range(max_m):
            if seeds:
                rank = seeds[(seed_rank + m) % len(seeds)]
                # rank is a factoradic RANK; decode directly.
                perm = factoradic_unrank(rank, self.n, signed=False)
            else:
                perm = np.arange(self.n)
            P   = APNPermuteOperator(self.n, perm)
            blk = P(self._base_block.astype(int)).astype(float)
            depth_blocks.append(blk)

        points  = np.zeros((num_points, self.d), dtype=np.float64)
        k_array = np.arange(num_points, dtype=np.int64)
        for m in range(max_m):
            v_m    = (k_array // (self.N ** m)) % self.N
            weight = 1.0 / (self.n ** (m + 1))
            points += depth_blocks[m][v_m] * weight

        return points

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"FractalNet(n={self.n}, d={self.d}, base_volume={self.N})"


# ── FractalNetKinetic ────────────────────────────────────────────────────────
#
# V15 addition: the TRUE FM-Dance digital net (T9 PROVEN).
# Uses path_coord (FM-Dance prefix-sum T matrix) instead of index_to_coords.
# FractalNet is kept unchanged as the experimental CONTROL GROUP.
#
# Theorem T9 (PROVEN, V15):
#   FractalNetKinetic is a LINEAR DIGITAL SEQUENCE with generator matrices C_m = T.
#   X_kin(k) = Σ_m (T · a_m(k) mod n) · n^{-(m+1)}
#   Since det(T) = -1 ≡ n-1 (mod n), T ∈ GL(d, Z_n): volume-preserving affine skew.
#   T belongs to the Pascal/binomial algebra underlying Faure sequences,
#   giving the same asymptotic discrepancy class O((log N)^d / N).
#   NOT a rank-1 lattice rule — that claim has been corrected.

class FractalNetKinetic(FractalNet):
    """
    Kinetic continuous space-filling digital net (T9 PROVEN, V15).

    Uses the true FM-Dance Hamiltonian prefix-sum traversal (path_coord)
    rather than the uncoupled addressing bijection (index_to_coords).

    Now uses TMatrixOperator and APNPermuteOperator as the Single Source of Truth.

    Mathematically this is a LINEAR DIGITAL SEQUENCE where the generator
    matrices are the FM-Dance lower-triangular prefix-sum matrix T:

        X_kin(k) = Σ_m  (T · a_m(k) mod n) · n^{-(m+1)}

    Since T is a linear operator independent of depth m, it factors out
    of the radical inverse sum:

        X_kin(k)  =  T · X_addr(k)   (digit-wise, mod n)

    Because det(T) = -1 mod n, T ∈ GL(d, Z_n), so this is a volume-
    preserving affine skew of the base digital net (FractalNet).

    FractalNet is the CONTROL GROUP — the uncoupled base-n van der Corput
    sequence. The difference in benchmark output directly measures the
    contribution of the T matrix.

    NOTE: The V14 dual-vector score of 0.000 at N=3^6 was a TRUNCATION
    ARTEFACT of the uncoupled net, not a FM-Dance property. See module
    docstring for the algebraic proof.

    Connection to Faure sequences (V15 finding):
        T is a degenerate Pascal matrix (all binomial weights = 1).
        Faure sequences use the full Pascal matrix. Both belong to the
        binomial transform family, so FractalNetKinetic inherits Faure-
        class discrepancy bounds up to a constant factor.

    Parameters
    ----------
    n : int   Radix. Must be odd (FM-Dance requirement).
    d : int   Spatial dimension (≥ 1).

    Examples
    --------
    >>> net_plain   = FractalNet(n=3, d=4)          # control: identity generator
    >>> net_kinetic = FractalNetKinetic(n=3, d=4)   # T-matrix generator
    >>> pts = net_kinetic.generate(81)
    >>> pts.shape
    (81, 4)
    """
    def __init__(self, n: int, d: int) -> None:
        super().__init__(n, d)
        
        # 1. Instantiate the T-Matrix Operator (The Bedrock Physics)
        # RefFinding #1: Fixed broadcast logic now lives here.
        self.T = TMatrixOperator(n)
        
        # 2. Vectorized Digit Extraction (The 'Digits' Stage)
        # Generate all digit vectors for the base block v ∈ [0, N)
        v_indices = np.arange(self.N, dtype=int)
        # shape (N, D)
        all_digits = np.array([index_to_coords(v, n, d) for v in v_indices]) + self.half
        
        # 3. Apply the Operator (The 'Integrate' Stage)
        # Φ_base = T · a
        kinetic_digits = self.T(all_digits) # Result is [0, n-1] unsigned
        
        # Store as base block for radical inverse
        self._base_block = kinetic_digits.astype(float)

    def generate_scrambled(
        self,
        num_points: int,
        seed_rank: int = 0,
        mode: str = "owen",
    ) -> np.ndarray:
        """
        Generate an APN-scrambled kinetic sequence (DN2).

        Dispatches to FLU-Owen (default) or coordinated mode.
        The kinetic base block already contains T-transformed digits
        (the FM-Dance prefix-sum), so the scrambling pipeline is:

            Digits → T-transform (cached in _base_block) → A_{m,[i]} → accumulate

        ``"owen"`` (default)
            Independent APN permutation per (depth, dimension) pair.
            Seed at (m, i): GOLDEN_SEEDS[n][(seed_rank + m*d + i) % len(seeds)].
            Matches Owen (1995) structural independence; enables asymptotic
            discrepancy constant improvement (DN2 closure path).

        ``"coordinated"``
            One APN permutation per depth, shared across dimensions (V15.1.3).
            Retained for backward compatibility.

        STATUS: DN2-P1, DN2-P2 PROVEN; DN2-P3 (FFT) CONFIRMED.
        Asymptotic L2 constant improvement OPEN — see
        docs/PROOF_DN2_APN_SCRAMBLING.md.

        Parameters
        ----------
        num_points : int
        seed_rank  : int   base index into GOLDEN_SEEDS[n]
        mode       : str   ``"owen"`` (default) or ``"coordinated"``

        Returns
        -------
        np.ndarray  shape (num_points, d)  values in [0, 1)
        """
        if mode == "owen":
            return self.generate_owen_scrambled(num_points, seed_rank=seed_rank)
        elif mode == "coordinated":
            return self._generate_coordinated_scrambled(num_points, seed_rank=seed_rank)
        else:
            raise ValueError(f"mode must be 'owen' or 'coordinated', got {mode!r}")

    def generate_owen_scrambled(
        self,
        num_points: int,
        seed_rank: int = 0,
    ) -> np.ndarray:
        """
        FLU-Owen scrambling on T-transformed digits: independent APN per (depth, dim).

        Applies a distinct APN permutation A_{m,i} to each coordinate i
        independently at each depth m.  The kinetic _base_block contains
        T-transformed digits (FM-Dance prefix-sum), so this implements the
        full DN2 pipeline:

            a_m(k)  →  T·a_m  (cached, T9 PROVEN)
                    →  A_{m,i} per dimension  (FLU-Owen, DN2-P1/P2 PROVEN)
                    →  accumulate at n^{-(m+1)}

        Parameters
        ----------
        num_points : int
        seed_rank  : int   base index; (m,i) uses (seed_rank + m*d + i) mod |seeds|

        Returns
        -------
        np.ndarray  shape (num_points, d)  values in [0, 1)
        """
        from flu.core.factoradic import factoradic_unrank, GOLDEN_SEEDS

        seeds = GOLDEN_SEEDS.get(self.n, [])

        if num_points <= 0:
            return np.zeros((0, self.d), dtype=float)

        max_m = 1
        while self.N ** max_m <= num_points:
            max_m += 1

        depth_blocks = []
        base_int = self._base_block.astype(int)
        for m in range(max_m):
            blk = base_int.copy()                              # shape (N_base, d)
            for i in range(self.d):
                if seeds:
                    seed_idx = (seed_rank + m * self.d + i) % len(seeds)
                    # seeds[seed_idx] is a factoradic RANK; decode directly.
                    perm = factoradic_unrank(seeds[seed_idx], self.n, signed=False)
                else:
                    perm = np.arange(self.n)
                blk[:, i] = perm[blk[:, i]]
            depth_blocks.append(blk.astype(float))

        points  = np.zeros((num_points, self.d), dtype=float)
        k_array = np.arange(num_points, dtype=np.int64)
        for m in range(max_m):
            v_m    = (k_array // (self.N ** m)) % self.N
            weight = 1.0 / (self.n ** (m + 1))
            points += depth_blocks[m][v_m] * weight

        return points

    def _generate_coordinated_scrambled(
        self,
        num_points: int,
        seed_rank: int = 0,
    ) -> np.ndarray:
        """
        Coordinated scrambling on T-transformed digits (V15.1.3 legacy).

        One APN permutation per depth applied to all D dimensions together.
        Retained for backward compatibility and V15 benchmark reproducibility.
        Prefer generate_owen_scrambled() for new work.
        """
        from flu.core.factoradic import factoradic_unrank, GOLDEN_SEEDS

        seeds = GOLDEN_SEEDS.get(self.n, [])

        if num_points <= 0:
            return np.zeros((0, self.d), dtype=float)

        max_m = 1
        while self.N ** max_m <= num_points:
            max_m += 1

        depth_blocks = []
        for m in range(max_m):
            rank = seeds[(seed_rank + m) % len(seeds)] if seeds else 0
            # rank is a factoradic RANK; decode directly.
            perm = factoradic_unrank(rank, self.n, signed=False)
            P    = APNPermuteOperator(self.n, perm)
            blk  = P(self._base_block.astype(int)).astype(float)
            depth_blocks.append(blk)

        points  = np.zeros((num_points, self.d), dtype=float)
        k_array = np.arange(num_points, dtype=np.int64)
        for m in range(max_m):
            v_m    = (k_array // (self.N ** m)) % self.N
            weight = 1.0 / (self.n ** (m + 1))
            points += depth_blocks[m][v_m] * weight

        return points

    def __repr__(self) -> str:
        return (
            f"FractalNetKinetic(n={self.n}, d={self.d}, "
            f"base_volume={self.N}, T_matrix=True)"
        )
