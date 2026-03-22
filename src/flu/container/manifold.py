"""
flu/container/manifold.py
=========================
Sparse Manifold Address — bijection between the 9×9 HyperCell grid
and the 3⁴ FM-Dance torus.

Single Responsibility: translate between two coordinate systems without
knowing anything about UKMC contracts or application logic.

THEOREM (Unification / Seam), STATUS: PROVEN — see verify_seam().

The key insight:
    A cell at grid position (row, col) has norm0 = (d1-1)*9 + (d2-1) ∈ [0,80].
    norm0 is a valid FM-Dance step index for the 3⁴ torus (n=3, d=4).
    Therefore: cell.norm0 ↔ index_to_coords(norm0, 3, 4) is a bijection.

Dependencies: flu.core.fm_dance, flu.core.lo_shu.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

from flu.core.fm_dance import index_to_coords, coords_to_index
from flu.core.lo_shu   import LoShuHyperCell, CellStrata


# ── Constants ─────────────────────────────────────────────────────────────────

_N = 3
_D = 4
_TOTAL_CELLS = _N ** _D   # 81


# ── Address functions ─────────────────────────────────────────────────────────

def cell_to_sparse_coords(cell: CellStrata) -> Tuple[int, ...]:
    """
    Map a HyperCell cell to its 4D FM-Dance sparse coordinate.

    Parameters
    ----------
    cell : CellStrata

    Returns
    -------
    coords : tuple of 4 ints ∈ {-1, 0, 1}

    Complexity
    ----------
    O(d) = O(4)
    """
    return index_to_coords(cell.norm0, _N, _D)


def sparse_coords_to_norm0(coords: Tuple[int, ...]) -> int:
    """
    Map a 4D FM-Dance coordinate to norm0 (HyperCell step index).

    Parameters
    ----------
    coords : tuple of 4 ints ∈ {-1, 0, 1}

    Returns
    -------
    norm0 : int ∈ [0, 80]

    Complexity
    ----------
    O(d) = O(4)
    """
    return coords_to_index(coords, _N, _D)


def cell_at_sparse_coords(
    hc    : LoShuHyperCell,
    coords: Tuple[int, ...],
) -> CellStrata:
    """
    Retrieve the CellStrata at a given 4D sparse coordinate.

    Parameters
    ----------
    hc     : LoShuHyperCell
    coords : tuple of 4 ints ∈ {-1, 0, 1}

    Returns
    -------
    CellStrata

    Raises
    ------
    ValueError  if coords do not resolve to a valid grid position
    """
    norm0 = sparse_coords_to_norm0(coords)
    norm1 = norm0 + 1
    pos   = hc.address_of(norm1)
    if pos is None:
        raise ValueError(
            f"norm1={norm1} (from coords={coords}) not found in HyperCell grid."
        )
    return hc.cell(*pos)


# ── Seam verification ─────────────────────────────────────────────────────────

def verify_seam(hc: LoShuHyperCell) -> Dict[str, Any]:
    """
    Verify the FM-Dance ↔ HyperCell seam for all 81 cells.

    THEOREM (Unification / Seam), STATUS: PROVEN
    ─────────────────────────────────────────────
    Statement:
        For every norm0 ∈ [0, 80]:
            (a) coords_to_index(index_to_coords(norm0, 3, 4), 3, 4) = norm0
            (b) The cell at the grid position of norm1 = norm0+1 has
                cell_to_sparse_coords(cell) == index_to_coords(norm0, 3, 4).

    Proof:
        (a) is the FM-Dance bijection round-trip (Theorem in fm_dance.py).
        (b) follows because norm0 = (d1-1)*9 + (d2-1) was the FM-Dance step
            index by construction; the grid encodes exactly this index in
            the norm0 field.  □

    Parameters
    ----------
    hc : LoShuHyperCell  (any perspective)

    Returns
    -------
    dict with keys: round_trip_errors, seam_errors, seam_verified
    """
    rt_errors   = 0
    seam_errors = 0

    for norm0 in range(_TOTAL_CELLS):
        # (a) Round-trip
        coords  = index_to_coords(norm0, _N, _D)
        norm0_r = coords_to_index(coords, _N, _D)
        if norm0_r != norm0:
            rt_errors += 1

        # (b) Seam
        norm1 = norm0 + 1
        pos   = hc.address_of(norm1)
        if pos is None:
            seam_errors += 1
            continue
        cell        = hc.cell(*pos)
        coords_back = cell_to_sparse_coords(cell)
        norm0_back  = coords_to_index(coords_back, _N, _D)
        if norm0_back != norm0:
            seam_errors += 1

    return {
        "round_trip_errors": rt_errors,
        "seam_errors"      : seam_errors,
        "seam_verified"    : rt_errors == 0 and seam_errors == 0,
    }
