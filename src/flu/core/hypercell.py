"""
flu/core/hypercell.py
=====================
FLUHyperCell — main user-facing cell class.

Single Responsibility: *orchestrate* Lo Shu geometry, UKMC contract,
and sparse manifold addressing.  Delegates all maths to its components.

The `set_perspective()` method uses a dedicated
`_sync_contract_to_perspective()` helper so contract mutation is
intentional and traceable, and the contract is only mutated if not yet
frozen.


Dependencies: flu.core.lo_shu, flu.container.contract,
              flu.container.manifold, flu.core.fm_dance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flu.core.lo_shu     import LoShuHyperCell, Perspective, CellStrata, _LO_SHU_NP
from flu.container.contract  import UKMCContract
from flu.container.manifold  import (
    cell_to_sparse_coords,
    cell_at_sparse_coords,
    verify_seam,
)
from flu.core.fm_dance   import index_to_coords


class FLUHyperCell:
    """
    3⁴ Fractal Unit Cell with UKMC contract and sparse manifold addressing.

    Extends LoShuHyperCell (core.lo_shu) with:
      - UKMCContract  (τ, Λ, Ω, Φ, Δ, ⊗)
      - SparseManifold bridge  (norm0 ↔ 4D FM-Dance coords)
      - Pivot-constrained cell selection
      - Perspective control with optional contract sync

    Parameters
    ----------
    perspective      : Perspective   default = identity (0,0,0)
    container_weight : float         Wc / Ω.  Default = 1.0.
    tau              : int           τ fractal depth.   Default = 0.
    logos            : dict | None   Λ genesis seed.    Default = auto.
    lo_shu           : np.ndarray | None  3×3 seed.     Default = canonical.
    """

    def __init__(
        self,
        perspective     : Optional[Perspective]  = None,
        container_weight: float                   = 1.0,
        tau             : int                     = 0,
        logos           : Optional[Dict]          = None,
        lo_shu          : Optional[np.ndarray]    = None,
    ) -> None:
        self._perspective = perspective or Perspective()
        self._lo_shu_seed = (
            lo_shu if lo_shu is not None else _LO_SHU_NP
        ).copy().astype(np.int64)

        # Inner Lo Shu geometry engine
        self._hc = LoShuHyperCell(
            perspective      = self._perspective,
            container_weight = container_weight,
            lo_shu           = self._lo_shu_seed,
        )

        # UKMC contract
        _logos = logos or {
            "method"    : "lo_shu_sudoku",
            "n"         : 3,
            "d"         : 4,
            "phase_idx" : self._perspective.phase_idx,
            "shift_r"   : self._perspective.shift_r,
            "shift_c"   : self._perspective.shift_c,
            "lo_shu"    : self._lo_shu_seed.tolist(),
        }
        self.contract = UKMCContract(
            tau   = tau,
            logos = _logos,
            omega = container_weight,
            phi   = {
                "phase_idx": self._perspective.phase_idx,
                "shift_r"  : self._perspective.shift_r,
                "shift_c"  : self._perspective.shift_c,
            },
        )

    # ── Delegation to inner Lo Shu geometry ───────────────────────────────

    def cell(self, row: int, col: int) -> CellStrata:
        """CellStrata at grid position (row, col)."""
        return self._hc.cell(row, col)

    def center(self) -> CellStrata:
        """Centre cell (row=4, col=4)."""
        return self._hc.center()

    def balanced(self) -> np.ndarray:
        """9×9 balanced-value array."""
        return self._hc.balanced()

    def norm0(self) -> np.ndarray:
        """9×9 zero-based index array."""
        return self._hc.norm0()

    def norm1(self) -> np.ndarray:
        """9×9 one-based index array."""
        return self._hc.norm1()

    def unity(self) -> np.ndarray:
        """9×9 unity-value array."""
        return self._hc.unity()

    def gnosis(self) -> np.ndarray:
        """G_cell = unity × Ω  (contract-weighted unity)."""
        return self._hc.unity() * self.contract.omega

    def address_of(self, n1_value: int) -> Optional[Tuple[int, int]]:
        """Find grid (row, col) for a given norm1 value.  O(1) lookup."""
        return self._hc.address_of(n1_value)

    # ── Sparse manifold interface ──────────────────────────────────────────

    def sparse_address(self, row: int, col: int) -> Tuple[int, ...]:
        """
        Grid position (row, col) → 4D FM-Dance sparse coordinate.
        O(d) = O(4) computation.
        """
        cell = self._hc.cell(row, col)
        return cell_to_sparse_coords(cell)

    def cell_at_sparse(self, coords: Tuple[int, ...]) -> CellStrata:
        """
        4D FM-Dance coordinate → CellStrata.
        O(d) = O(4) lookup.
        """
        return cell_at_sparse_coords(self._hc, coords)

    def sparse_step_index(self, row: int, col: int) -> int:
        """norm0 (FM-Dance step index) of cell at (row, col)."""
        return self._hc.cell(row, col).norm0

    # ── Pivot-constrained cell selection ──────────────────────────────────

    def cells_with_pivot(
        self,
        pivot_value: int,
        dimension  : int = 0,
    ) -> List[Tuple[int, int]]:
        """
        Return all (row, col) positions whose 4D sparse address has
        coords[dimension] == pivot_value.

        THEOREM (Lo Shu Centre Constraint):
            The centre cell (4,4) has norm0=40, giving sparse coords
            index_to_coords(40, 3, 4).  Setting pivot_value=0 and
            dimension=0 selects the "centre plane" of the torus.

        Parameters
        ----------
        pivot_value : int ∈ {-1, 0, 1}
        dimension   : int ∈ {0, 1, 2, 3}
        """
        if pivot_value not in (-1, 0, 1):
            raise ValueError(f"pivot_value must be in {{-1,0,1}}, got {pivot_value}")
        if not (0 <= dimension < 4):
            raise ValueError(f"dimension must be in [0,3], got {dimension}")

        n1_grid = self._hc.norm1()
        result  = []
        for r in range(9):
            for c in range(9):
                norm0  = int(n1_grid[r, c]) - 1
                coords = index_to_coords(norm0, 3, 4)
                if coords[dimension] == pivot_value:
                    result.append((r, c))
        return result

    # ── Perspective control ────────────────────────────────────────────────

    def set_perspective(self, p: Perspective) -> "FLUHyperCell":
        """
        Change the geometric perspective and optionally sync the contract.

        If the contract is frozen, phi and logos are NOT updated
        (an already-frozen contract records the original identity).
        """
        self._perspective = p
        self._hc.set_perspective(p)
        if not self.contract.is_frozen:
            self._sync_contract_to_perspective(p)
        return self

    def _sync_contract_to_perspective(self, p: Perspective) -> None:
        """Update the contract's Φ and Λ fields to match perspective p."""
        self.contract.phi = {
            "phase_idx": p.phase_idx,
            "shift_r"  : p.shift_r,
            "shift_c"  : p.shift_c,
        }
        self.contract.logos["phase_idx"] = p.phase_idx
        self.contract.logos["shift_r"]   = p.shift_r
        self.contract.logos["shift_c"]   = p.shift_c

    # ── UKMC helpers ──────────────────────────────────────────────────────

    def set_omega(self, wc: float) -> "FLUHyperCell":
        """Update gnostic weight Ω and propagate to inner HyperCell."""
        if not self.contract.is_frozen:
            self.contract.omega = wc
        self._hc.container_weight = wc
        return self

    def zoom_in(self) -> "FLUHyperCell":
        """Increment τ (zoom into finer fractal level)."""
        self.contract.tau += 1
        return self

    def zoom_out(self) -> "FLUHyperCell":
        """Decrement τ (zoom out to coarser fractal level)."""
        self.contract.tau -= 1
        return self

    # ── Verification ──────────────────────────────────────────────────────

    def verify(self, silent: bool = True) -> Dict[str, Any]:
        """
        Run all structural checks plus seam and contract validation.

        Returns a unified result dict.
        """
        v8_result   = self._hc.verify(silent=silent)
        seam_result = verify_seam(self._hc)

        contract_ok = (
            len(self.contract.logos) > 0
            and self.contract.omega > 0
        )

        result = {
            **v8_result,
            **seam_result,
            "contract_ok"  : contract_ok,
            "fully_verified": (
                v8_result.get("verified", False)
                and seam_result["seam_verified"]
                and contract_ok
            ),
            # Backward-compat keys
            "contract_ok"     : contract_ok,
            "fully_verified"  : (
                v8_result.get("verified", False)
                and seam_result["seam_verified"]
                and contract_ok
            ),
        }

        if not silent:
            status = "✓ VERIFIED" if result["fully_verified"] else "✗ FAILED"
            print(f"FLUHyperCell [{self._perspective}]: {status}")

        return result

    # ── Fractal embedding ─────────────────────────────────────────────────

    def embed_as_3_6(
        self,
        micro_lo_shu: "Optional[np.ndarray]" = None,
    ) -> "FractalHyperCell_3_6":
        """
        Embed this 3⁴ HyperCell as the macro layer of a 3⁶ FractalHyperCell.

        Returns a FractalHyperCell_3_6 where self is the macro and each
        macro cell contains a full 3×3 Lo Shu micro grid.

        Parameters
        ----------
        micro_lo_shu : np.ndarray | None   3×3 Lo Shu seed for micro layer.
                       Defaults to canonical Lo Shu if None.

        Returns
        -------
        FractalHyperCell_3_6

        STATUS: PROVEN — tensor product of two proven bijections (see fractal_3_6.py).
        """
        # Import here to keep hypercell.py dependency-order correct;
        # fractal_3_6 is a layer above hypercell.
        from flu.core.fractal_3_6 import FractalHyperCell_3_6
        return FractalHyperCell_3_6(macro=self, micro_lo_shu=micro_lo_shu)

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FLUHyperCell("
            f"perspective={self._perspective}, "
            f"Ω={self.contract.omega:.3f}, "
            f"τ={self.contract.tau}, "
            f"id={self.contract.identity_hash()[:8]}…)"
        )


