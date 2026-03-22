"""
Tests for flu.container.manifold — sparse manifold seam.

THEOREM (Unification / Seam): for all 81 cells, the round-trip
    coords_to_index(index_to_coords(norm0)) == norm0
and the seam between HyperCell and FM-Dance is exact.
"""

import pytest
from flu.core.lo_shu      import LoShuHyperCell, Perspective
from flu.container.manifold import (
    cell_to_sparse_coords,
    sparse_coords_to_norm0,
    cell_at_sparse_coords,
    verify_seam,
)


class TestSeamVerification:
    def test_default_perspective(self):
        hc     = LoShuHyperCell()
        result = verify_seam(hc)
        assert result["seam_verified"], (
            f"Seam verification failed: {result}"
        )

    @pytest.mark.parametrize("pid", [0, 1, 8, 35, 71])
    def test_all_sampled_perspectives(self, pid):
        p      = Perspective.from_id(pid)
        hc     = LoShuHyperCell(perspective=p)
        result = verify_seam(hc)
        assert result["seam_verified"], (
            f"Seam failed for perspective id={pid}: {result}"
        )


class TestCellToCoords:
    def test_center_cell_norm0(self):
        """Centre cell norm0=40 → known coords for index_to_coords(40,3,4)."""
        from flu.core.fm_dance import index_to_coords
        hc   = LoShuHyperCell()
        cell = hc.center()
        assert cell.norm0 == 40
        coords = cell_to_sparse_coords(cell)
        expected = index_to_coords(40, 3, 4)
        assert coords == expected

    def test_all_81_cells_have_unique_coords(self):
        hc      = LoShuHyperCell()
        coords_set = set()
        for r in range(9):
            for c in range(9):
                cell   = hc.cell(r, c)
                coords = cell_to_sparse_coords(cell)
                assert len(coords) == 4
                coords_set.add(coords)
        assert len(coords_set) == 81, "Expected 81 unique sparse addresses"


class TestRoundTrip:
    def test_norm0_round_trip(self):
        for norm0 in range(81):
            from flu.core.fm_dance import index_to_coords, coords_to_index
            coords = index_to_coords(norm0, 3, 4)
            back   = sparse_coords_to_norm0(coords)
            assert back == norm0, f"Round-trip failed for norm0={norm0}"

    def test_cell_at_sparse_round_trip(self):
        hc = LoShuHyperCell()
        for r in range(9):
            for c in range(9):
                cell_orig = hc.cell(r, c)
                coords    = cell_to_sparse_coords(cell_orig)
                cell_back = cell_at_sparse_coords(hc, coords)
                assert cell_back.norm0 == cell_orig.norm0
