"""
src/flu/interfaces/hilbert.py
==============================
HilbertFacet — FM-Dance Tuned for L2-Clustering / Space-Filling Curves (HIL-1).

⚠️  DEPRECATED / RETIRED — V15.1.3
  HIL-1 has been retired. See theorem registry entry HIL-1 (status: RETIRED).

  Root cause of retirement:
    The HIL-1 conjecture named n=2 (binary) as its primary case and cited
    evidence only at d=2, n=2. However, FM-Dance requires odd n, and this
    constructor enforces that with a ValueError for even n. The stated primary
    case is therefore forbidden by the implementation — a fundamental
    self-contradiction. At valid odd n (n=3, 5, 7, …), no locality improvement
    over plain FM-Dance was ever confirmed in any benchmark.

  This file is retained as a research artifact. The RotationHub idea
  (carry-level hyperoctahedral rotations) is noted in docs/ROADMAP.md as a
  future direction under a corrected odd-n framing.

  For production traversals: use flu.core.fm_dance_path.path_coord or
  flu.interfaces.gray_code.GrayCodeFacet.

Original description (ARCHIVED):
  HilbertFacet — FM-Dance Tuned for L2-Clustering / Space-Filling Curves (HIL-1).
  Status: CONJECTURE (HIL-1)
    Not promoted from CONJECTURE without formal proof that RotationHub
    achieves true Hilbert L2 clustering. Current evidence is computational
    (qualitative locality improvement at d=2, n=2).

  Mathematical framing:
    The standard Hilbert curve construction applies recursive rotations
    and reflections at each quadrant level ("carry level" in FLU terms).
    HIL-1 conjectures that the FLU FM-Dance traversal, augmented with the
    RotationHub hyperoctahedral-group action triggered at carry levels,
    approximates the L2-clustering property of Hilbert curves.

    Formally:
      Let Φ_H(k) = RotationHub(path_coord(k, n, d), j(k), n)
      where j(k) = ν_n(k) is the carry depth (n-adic valuation).

    Conjecture: For n=2, Φ_H generates coordinates whose L2-star
    discrepancy is ≤ that of plain FM-Dance, and spatial locality
    (adjacent ranks → adjacent coordinates) is improved.

  Open path to PROVEN status (archived — conjecture retired):
    1. Formally define the hyperoctahedral group H_D action on Z_n^D.
    2. Prove the tuned path is still a Hamiltonian (visits all n^D points).
    3. Measure D* and locality vs standard FM-Dance and true Hilbert curve.
    4. Bound the L2 error analytically (requires Hilbert curve theory).

V15 — audit integration sprint.
V15.1.3 — HIL-1 RETIRED.
"""

from __future__ import annotations

import warnings as _warnings

from typing import Optional

import numpy as np

from flu.interfaces.base import FluFacet
from flu.core.fm_dance_path import path_coord


class RotationHub:
    """
    Hyperoctahedral group actions (signed permutations) triggered at carry levels.

    At carry level j (when j lower digits of rank k all equal n-1),
    applies a cyclic coordinate shift of j positions and a parity-dependent
    toroidal reflection to emulate the recursive quadrant orientation of
    Hilbert-curve construction.
    """

    def __init__(self, d: int) -> None:
        self.d = d

    def apply_at_carry(self, coords: np.ndarray, j: int, n: int) -> np.ndarray:
        """
        Apply the hyperoctahedral rotation for carry level j.

        Parameters
        ----------
        coords : ndarray, shape (d,)
            Current coordinate vector in Z_n^d.
        j : int
            Carry level (0 = no carry; d = maximum carry depth).
        n : int
            Radix.

        Returns
        -------
        ndarray, shape (d,) — rotated coordinate vector.
        """
        # Cyclic axis shift by j
        res = np.roll(coords, j)
        # Parity-dependent reflection to satisfy Hilbert U-shape orientation
        if j % 2 != 0:
            res = (n - 1) - res  # toroidal reflection
        return res


class HilbertFacet(FluFacet):
    """
    FM-Dance tuned for L2 clustering via carry-level RotationHub (HIL-1).

    CONJECTURE — not proven. Use plain FM-Dance (traverse / path_coord)
    for production traversals. This facet is provided for research into
    Hilbert-like space-filling properties.

    Parameters
    ----------
    d : int
        Dimension.
    n : int
        Radix. Best results at n=2 (binary, matching Hilbert quadrant split).
    tune : bool
        If True (default), apply RotationHub at carry boundaries.
        If False, degenerates to plain FM-Dance.

    Examples
    --------
    >>> hf = HilbertFacet(d=2, n=2)
    >>> pt = hf.get_point(k=5)
    >>> all_pts = hf.get_all_points()
    """

    def __init__(self, d: int, n: int = 3, tune: bool = True) -> None:
        if n % 2 == 0:
            raise ValueError(
                f"HilbertFacet requires odd n (FM-Dance constraint), got n={n}. "
                "Use n=3 for a ternary Hilbert-like curve."
            )
        _warnings.warn(
            "HilbertFacet (HIL-1) is RETIRED as of V15.1.3. "
            "The n=2 primary case is forbidden by FM-Dance's odd-n requirement, "
            "making the conjecture's evidence base inaccessible. "
            "No locality improvement was confirmed for valid odd n. "
            "For production traversals use flu.core.fm_dance_path.path_coord or GrayCodeFacet. "
            "See docs/ROADMAP.md for the RotationHub research direction.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            name="HilbertFacet",
            theorem_id="HIL-1",
            status="RETIRED",
            description=(
                "RETIRED (V15.1.3). FM-Dance + RotationHub was conjectured to approximate "
                "Hilbert L2-clustering but the primary case (n=2) is forbidden by FM-Dance. "
                "See docs/ROADMAP.md for future odd-n research direction."
            ),
        )
        self.d = d
        self.n = n
        self.tune = tune
        self.hub = RotationHub(d)

    # ── core access ───────────────────────────────────────────────────────────

    def get_point(self, k: int) -> np.ndarray:
        """
        Return the k-th traversal point with Hilbert-style rotation applied.

        O(D) time (single path_coord call + O(D) rotation).
        Coordinates are unsigned (in [0, n)).

        Parameters
        ----------
        k : int
            Rank in [0, n^d).

        Returns
        -------
        ndarray, shape (d,) of ints in [0, n).
        """
        raw = path_coord(k, self.n, self.d)
        # Normalise signed coords to unsigned [0, n)
        coords = np.array([int(c) % self.n for c in raw])
        if self.tune:
            # Carry-detection: count how many low-order digits of k equal n-1
            j = 0
            temp_k = k
            while j < self.d and temp_k % self.n == self.n - 1:
                temp_k //= self.n
                j += 1
            if j > 0:
                coords = self.hub.apply_at_carry(coords, j, self.n)
        return coords

    def get_all_points(self) -> np.ndarray:
        """
        Return all n^d traversal points as an array of shape (n^d, d).

        NOTE (CONJECTURE): The tuned path is not guaranteed to be a
        Hamiltonian path on Z_n^D (all-points-visited property is open
        for the tuned case). Verify with check_hamiltonian() before use.
        """
        N = self.n ** self.d
        return np.array([self.get_point(k) for k in range(N)])

    def check_hamiltonian(self) -> bool:
        """
        Verify the tuned path visits every point exactly once.

        Returns True iff the path is a valid Hamiltonian traversal.
        """
        pts = self.get_all_points()
        seen = set(map(tuple, pts.tolist()))
        N = self.n ** self.d
        return len(seen) == N

    def locality_score(self) -> float:
        """
        Compute a locality score: mean L2 distance between consecutive points.

        Lower = better locality (Hilbert curve has provably good locality).
        Compare to plain FM-Dance for reference.
        """
        pts = self.get_all_points().astype(float)
        diffs = pts[1:] - pts[:-1]
        return float(np.mean(np.linalg.norm(diffs, axis=1)))
