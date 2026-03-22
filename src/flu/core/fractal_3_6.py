"""
flu/core/fractal_3_6.py
=======================
FractalHyperCell_3_6 — recursive 3⁶ embedding.

TENSOR PRODUCT STRUCTURE, STATUS: PROVEN
─────────────────────────────────────────
Each of the 81 macro cells (FLUHyperCell, 3⁴ torus) contains a full
Lo Shu 3×3 micro grid (3² torus).  Total 729 = 3⁶ cells, each with a
unique 6D sparse manifold address in {-1, 0, 1}⁶.

Proof sketch:
  • Macro bijection:  grid pos (r,c) ↔ 4D coord  — proven in manifold.py
  • Micro bijection:  lo_shu[r,c]−1 = norm0 ∈ [0,8]
                      ↔ index_to_coords(norm0, 3, 2) ∈ {-1,0,1}²
                      — proven (FM-Dance bijection, fm_dance.py)
  • Product:  (3⁴ bijection) × (3² bijection) → 3⁶ bijection
              Cartesian product of two bijections is a bijection.  □

Address layout:  coords_6d = macro_4d + micro_2d
    coords_6d[0:4]  ←  FLUHyperCell.sparse_address(macro_row, macro_col)
    coords_6d[4:6]  ←  index_to_coords(lo_shu[micro_row, micro_col]−1, 3, 2)

Dependencies: flu.core.lo_shu, flu.core.fm_dance, flu.core.hypercell.
No new external deps beyond numpy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from flu.core.lo_shu    import _LO_SHU_NP, CellStrata
from flu.core.fm_dance  import index_to_coords, coords_to_index
from flu.core.hypercell import FLUHyperCell


# ── Micro cell result type ────────────────────────────────────────────────────

class MicroCell:
    """
    Minimal descriptor for one cell in the 3×3 Lo Shu micro grid.

    Attributes
    ----------
    row, col   : int   0-based position in the 3×3 micro grid
    value      : int   canonical Lo Shu value ∈ {1, …, 9}
    norm0      : int   zero-based index = value − 1  ∈ [0, 8]
    coords_2d  : (int, int)   FM-Dance 2D sparse address ∈ {-1,0,1}²
    """
    __slots__ = ("row", "col", "value", "norm0", "coords_2d")

    def __init__(self, row: int, col: int, value: int) -> None:
        self.row       = row
        self.col       = col
        self.value     = value
        self.norm0     = value - 1
        self.coords_2d: Tuple[int, int] = index_to_coords(self.norm0, 3, 2)

    def __repr__(self) -> str:
        return (
            f"MicroCell(row={self.row}, col={self.col}, "
            f"value={self.value}, coords_2d={self.coords_2d})"
        )


class CellPair:
    """
    Paired result of FractalHyperCell_3_6.cell_at_6d().

    Attributes
    ----------
    macro : CellStrata   the 3⁴ macro cell
    micro : MicroCell    the 3² micro cell
    """
    __slots__ = ("macro", "micro")

    def __init__(self, macro: CellStrata, micro: MicroCell) -> None:
        self.macro = macro
        self.micro = micro

    def __repr__(self) -> str:
        return f"CellPair(macro={self.macro}, micro={self.micro})"


# ── FractalHyperCell_3_6 ──────────────────────────────────────────────────────

class FractalHyperCell_3_6:
    """
    3⁶ Recursive Fractal Embedding: FLUHyperCell (3⁴) × Lo Shu (3²).

    Each of the 81 macro cells (9×9 grid) contains a full 3×3 Lo Shu
    micro grid.  Every one of the 729 resulting (macro, micro) pairs
    receives a unique 6D sparse address in {-1,0,1}⁶.

    Address layout
    --------------
        coords_6d[0:4]  macro 4D — from FLUHyperCell.sparse_address()
        coords_6d[4:6]  micro 2D — from index_to_coords(lo_shu_value−1, 3, 2)

    Parameters
    ----------
    macro        : FLUHyperCell   the 3⁴ macro layer (pre-built)
    micro_lo_shu : np.ndarray | None 3×3 seed for the micro grid.
                   Defaults to the canonical Lo Shu if None.

    STATUS: PROVEN — tensor product of two proven bijections.
    """

    TOTAL_CELLS   = 3 ** 6   # 729
    MACRO_N       = 9        # macro grid side length
    MICRO_N       = 3        # micro grid side length

    def __init__(
        self,
        macro       : FLUHyperCell,
        micro_lo_shu: Optional[np.ndarray] = None,
    ) -> None:
        self.macro         = macro
        self._micro_grid   = (
            micro_lo_shu if micro_lo_shu is not None else _LO_SHU_NP
        ).copy().astype(np.int64)

        if self._micro_grid.shape != (3, 3):
            raise ValueError(
                f"micro_lo_shu must be shape (3,3), got {self._micro_grid.shape}"
            )

        # Pre-build MicroCell objects (9, one per 3×3 position)
        self._micro_cells: Dict[Tuple[int, int], MicroCell] = {
            (r, c): MicroCell(r, c, int(self._micro_grid[r, c]))
            for r in range(3) for c in range(3)
        }

        # Build O(1) reverse index: 6D coords → (macro_row, macro_col, micro_row, micro_col)
        self._reverse: Dict[Tuple[int, ...], Tuple[int, int, int, int]] = {}
        self._build_index()

    # ── Index construction ────────────────────────────────────────────────

    def _build_index(self) -> None:
        """
        Populate self._reverse for all 729 (macro, micro) pairs.

        STATUS: PROVEN — follows from the bijection proof in the module docstring.
        Raises ValueError if any duplicate 6D address is detected.
        """
        for macro_r in range(self.MACRO_N):
            for macro_c in range(self.MACRO_N):
                macro_4d = self.macro.sparse_address(macro_r, macro_c)
                for micro_r in range(self.MICRO_N):
                    for micro_c in range(self.MICRO_N):
                        micro_2d  = self._micro_cells[(micro_r, micro_c)].coords_2d
                        coords_6d = macro_4d + micro_2d        # tuple concat

                        if coords_6d in self._reverse:
                            raise RuntimeError(
                                f"Duplicate 6D address {coords_6d} at "
                                f"macro=({macro_r},{macro_c}) "
                                f"micro=({micro_r},{micro_c})"
                            )
                        self._reverse[coords_6d] = (macro_r, macro_c, micro_r, micro_c)

    # ── Public address interface ──────────────────────────────────────────

    def sparse_address_6d(
        self,
        macro_row: int,
        macro_col: int,
        micro_row: int,
        micro_col: int,
    ) -> Tuple[int, ...]:
        """
        Grid positions → 6D FM-Dance sparse address.

        Parameters
        ----------
        macro_row, macro_col : int   position in 9×9 macro grid  [0, 8]
        micro_row, micro_col : int   position in 3×3 micro grid  [0, 2]

        Returns
        -------
        6-tuple ∈ {-1, 0, 1}⁶

        Complexity
        ----------
        O(d_macro + d_micro) = O(6)

        STATUS: PROVEN — direct composition of two proven bijections.
        """
        if not (0 <= macro_row < self.MACRO_N and 0 <= macro_col < self.MACRO_N):
            raise ValueError(
                f"macro position ({macro_row},{macro_col}) out of [0,{self.MACRO_N})"
            )
        if not (0 <= micro_row < self.MICRO_N and 0 <= micro_col < self.MICRO_N):
            raise ValueError(
                f"micro position ({micro_row},{micro_col}) out of [0,{self.MICRO_N})"
            )
        macro_4d = self.macro.sparse_address(macro_row, macro_col)
        micro_2d = self._micro_cells[(micro_row, micro_col)].coords_2d
        return macro_4d + micro_2d

    def cell_at_6d(self, coords_6d: Tuple[int, ...]) -> CellPair:
        """
        6D FM-Dance address → CellPair  (inverse lookup, O(1)).

        Parameters
        ----------
        coords_6d : 6-tuple ∈ {-1, 0, 1}⁶

        Returns
        -------
        CellPair  with .macro (CellStrata) and .micro (MicroCell)

        Raises
        ------
        ValueError  if coords_6d is not a valid 6D address

        Complexity
        ----------
        O(1) — cached reverse index built in __init__.

        STATUS: PROVEN — inverse of a proven bijection is a bijection.
        """
        pos = self._reverse.get(coords_6d)
        if pos is None:
            raise ValueError(f"6D address {coords_6d} not in index")
        macro_r, macro_c, micro_r, micro_c = pos
        return CellPair(
            macro=self.macro.cell(macro_r, macro_c),
            micro=self._micro_cells[(micro_r, micro_c)],
        )

    # ── Verification (ITER-3B seam check) ─────────────────────────────────

    def verify(self, silent: bool = True) -> Dict[str, Any]:
        """
        Verify the 3⁶ seam: all 729 cells have unique, valid 6D addresses
        and satisfy the round-trip property.

        THEOREM (FractalHyperCell_3_6 seam), STATUS: PROVEN
        ─────────────────────────────────────────────────────
        For every (macro_row, macro_col, micro_row, micro_col):
          (a) coords_6d ∈ {-1,0,1}⁶
          (b) All 729 coords_6d are distinct
          (c) cell_at_6d(coords_6d) round-trips back to the same positions

        Proof: (a) from digit-set bounds of index_to_coords with n=3.
               (b) follows from the bijection (module docstring proof).
               (c) follows from the cached index construction: both
                   forward and reverse maps are constructed together.  □

        Returns
        -------
        dict with keys:
            total_cells    : int   should be 729
            unique_addresses: int  should be 729
            range_errors   : int   should be 0
            round_trip_errors: int should be 0
            seam_verified  : bool
        """
        all_coords       = {}      # coords_6d → (mr, mc, ur, uc)
        range_errors     = 0
        round_trip_errors = 0

        for macro_r in range(self.MACRO_N):
            for macro_c in range(self.MACRO_N):
                for micro_r in range(self.MICRO_N):
                    for micro_c in range(self.MICRO_N):

                        coords = self.sparse_address_6d(
                            macro_r, macro_c, micro_r, micro_c
                        )

                        # (a) Range check — every coord must be in {-1, 0, 1}
                        if not all(c in (-1, 0, 1) for c in coords):
                            range_errors += 1

                        # (b) Uniqueness tracked via dict
                        all_coords[coords] = (macro_r, macro_c, micro_r, micro_c)

                        # (c) Round-trip: coords → positions → coords
                        pair        = self.cell_at_6d(coords)
                        pos_back    = self._reverse[coords]
                        coords_back = self.sparse_address_6d(*pos_back)
                        if coords_back != coords:
                            round_trip_errors += 1

        unique_count  = len(all_coords)
        total_count   = self.MACRO_N * self.MACRO_N * self.MICRO_N * self.MICRO_N
        seam_verified = (
            unique_count      == self.TOTAL_CELLS
            and range_errors  == 0
            and round_trip_errors == 0
        )

        result = {
            "total_cells"      : total_count,
            "unique_addresses" : unique_count,
            "range_errors"     : range_errors,
            "round_trip_errors": round_trip_errors,
            "seam_verified"    : seam_verified,
        }

        if not silent:
            status = "✓ 3⁶ SEAM VERIFIED" if seam_verified else "✗ 3⁶ SEAM FAILED"
            print(
                f"FractalHyperCell_3_6: {status} "
                f"({unique_count}/729 unique, "
                f"{range_errors} range errors, "
                f"{round_trip_errors} round-trip errors)"
            )

        return result

    # ── Convenience ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.TOTAL_CELLS

    def __repr__(self) -> str:
        return (
            f"FractalHyperCell_3_6("
            f"macro={self.macro!r}, "
            f"cells=729, "
            f"index_size={len(self._reverse)})"
        )
