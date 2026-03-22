"""
src/flu/interfaces/curves.py
============================
SpaceFillingCurveFacet — Generalized FM-Dance Toroidal Traversal.

STATUS: DESIGN_INTENT (research-grade bridge)

Mathematical Basis
──────────────────
  Generalises space-filling curves to the n-ary FM-Dance manifold.
  Mapping: Φ(k) = Ω_m(T · a(k) mod n)
  where Ω is a sequence of Hyperoctahedral H_D group actions (RotationHub).

Design choice
─────────────
  Rather than rigid Hilbert curve rotation tables (n=2 fixed), this facet 
  accepts an `FLUOperator` (the RotationHub) to allow researchers to define 
  their own recursive symmetry-breaking rules (Sierpiński/Peano/Hilbert-like).

Dependencies: flu.core.fm_dance_path, flu.core.operators, flu.interfaces.base.
"""

from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from flu.interfaces.base import FluFacet
from flu.core.fm_dance_path import path_coord
from flu.core.operators import RotationHubOperator


class CurveFacet(FluFacet):
    """
    Generalized FM-Dance Space-Filling Curve generator.

    Provides a recursive traversal of the toroidal lattice Z_n^D by applying
    a sequence of RotationHubOperators (H_D group actions) at carry boundaries.

    Parameters
    ----------
    n      : int   Radix (odd prime).
    d      : int   Spatial dimension.
    hub    : RotationHubOperator | None
             If provided, applies rotations at carry levels.
             If None, degenerates to pure FM-Dance (Identity operator).

    Examples
    --------
    >>> hub = RotationHubOperator(n=3, d=2, transition_rule=...)
    >>> facet = SpaceFillingCurveFacet(d=2, n=3, hub=hub)
    >>> pts = facet.generate(81)
    """

    def __init__(self, d: int, n: int = 3, hub: Optional[RotationHubOperator] = None) -> None:
        super().__init__(
            name="SpaceFillingCurveFacet",
            theorem_id="T8-GEN",
            status="RESEARCH",
            description=(
                "Generalized space-filling traversal on Z_n^D. "
                "Uses RotationHubOperator to apply recursive group actions "
                "on the FM-Dance foundation (T9)."
            ),
        )
        self.n = n
        self.d = d
        self.hub = hub

    def get_point(self, k: int) -> np.ndarray:
        """
        Generate the k-th point of the space-filling path in [0, 1)^D.
        Complexity: O(depth * D).
        """
        if self.hub:
            # 1. Recursive continuous evaluation via Hub (Fractal Algebra)
            digits = self._get_digit_hierarchy(k)
            return self.hub(digits)
            
        # 2. Fallback: Normalise the discrete FM-Dance coordinate into [0, 1)^D
        # This matches the m=1 output of FractalNetKinetic.
        raw = path_coord(k, self.n, self.d)
        
        # Convert signed [-half, half] to unsigned [0, n-1]
        half = self.n // 2
        unsigned_coords = np.array([int(c) + half for c in raw], dtype=float)
        
        # Project into the continuous unit hypercube
        return unsigned_coords / float(self.n)
        
    def generate(self, num_points: int) -> np.ndarray:
        """Generate first `num_points`."""
        return np.array([self.get_point(k) for k in range(num_points)])

    def _get_digit_hierarchy(self, k: int) -> List[np.ndarray]:
        """Convert k into a hierarchy of base-n digit vectors."""
        hierarchy = []
        rem = k
        for _ in range(self.d): # Depth scaling
            digits = []
            for _ in range(self.d):
                digits.append(rem % self.n)
                rem //= self.n
            hierarchy.append(np.array(digits))
            if rem == 0: break
        return hierarchy
