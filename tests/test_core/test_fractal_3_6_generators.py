"""
tests/test_core/test_fractal_3_6_generators.py
===============================================
Tests for the two FractalHyperCell_3_6 macro generators:
  - generator="sudoku"  (default, V15.3.1+)
  - generator="product" (legacy)
  - backward compat (explicit macro=FLUHyperCell())

The sudoku generator uses LoShuSudokuHyperCell as macro:
  macro 4D address = bt(d1) + bt(d2)  (Graeco-Latin BT, DN1-OA PROVEN)

The product generator uses FLUHyperCell as macro:
  macro 4D address = index_to_coords(norm0, 3, 4)  (FM-Dance)

Both produce 729 unique addresses in {-1,0,1}⁶.
"""

import pytest
import numpy as np

from flu.core.fractal_3_6 import (
    FractalHyperCell_3_6,
    SudokuMacroAdapter,
    CellPair,
    MicroCell,
)
from flu.core.hypercell    import FLUHyperCell
from flu.core.lo_shu       import CellStrata
from flu.core.lo_shu_sudoku import LoShuSudokuHyperCell


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frac_sudoku():
    return FractalHyperCell_3_6()                   # new default

@pytest.fixture(scope="module")
def frac_product():
    return FractalHyperCell_3_6(generator="product")

@pytest.fixture(scope="module")
def frac_explicit():
    return FractalHyperCell_3_6(FLUHyperCell())     # old explicit-macro style


# ── Default generator ─────────────────────────────────────────────────────────

class TestDefaultIsSudoku:
    def test_no_args_uses_sudoku(self, frac_sudoku):
        assert frac_sudoku.generator == "sudoku"

    def test_make_sudoku_factory(self):
        f = FractalHyperCell_3_6.make_sudoku()
        assert f.generator == "sudoku"

    def test_make_product_factory(self):
        f = FractalHyperCell_3_6.make_product()
        assert f.generator == "product"

    def test_generator_kwarg_sudoku(self):
        f = FractalHyperCell_3_6(generator="sudoku")
        assert f.generator == "sudoku"

    def test_generator_kwarg_product(self):
        f = FractalHyperCell_3_6(generator="product")
        assert f.generator == "product"

    def test_explicit_macro_overrides_generator(self):
        # Explicit FLUHyperCell always = product behaviour
        f = FractalHyperCell_3_6(FLUHyperCell())
        assert f.generator == "product"

    def test_sudoku_hypercell_property(self, frac_sudoku):
        hc = frac_sudoku.sudoku_hypercell
        assert isinstance(hc, LoShuSudokuHyperCell)

    def test_sudoku_hypercell_none_for_product(self, frac_product):
        assert frac_product.sudoku_hypercell is None


# ── SudokuMacroAdapter ────────────────────────────────────────────────────────

class TestSudokuMacroAdapter:
    def test_sparse_address_returns_4_tuple(self):
        a = SudokuMacroAdapter()
        addr = a.sparse_address(0, 0)
        assert len(addr) == 4

    def test_sparse_address_in_digit_set(self):
        a = SudokuMacroAdapter()
        for r in range(9):
            for c in range(9):
                addr = a.sparse_address(r, c)
                assert all(x in (-1, 0, 1) for x in addr), \
                    f"({r},{c}) addr {addr} out of {{-1,0,1}}"

    def test_81_unique_addresses(self):
        a = SudokuMacroAdapter()
        addrs = {a.sparse_address(r, c) for r in range(9) for c in range(9)}
        assert len(addrs) == 81, \
            f"Only {len(addrs)} unique macro addresses, expected 81"

    def test_all_81_tuples_covered(self):
        """OA(81,4,3,4): every 4-tuple in {-1,0,1}^4 appears exactly once."""
        import itertools
        a = SudokuMacroAdapter()
        addrs = {a.sparse_address(r, c) for r in range(9) for c in range(9)}
        all_tuples = set(itertools.product((-1, 0, 1), repeat=4))
        assert addrs == all_tuples, \
            "Sudoku macro addresses do not cover all of {-1,0,1}^4"

    def test_cell_returns_cell_strata(self):
        a = SudokuMacroAdapter()
        c = a.cell(4, 4)
        assert isinstance(c, CellStrata)

    def test_centre_cell_is_balanced_zero(self):
        a = SudokuMacroAdapter()
        c = a.cell(4, 4)
        assert c.balanced == 0

    def test_centre_address_is_all_zeros(self):
        """Centre (4,4) has d1=5, d2=5 → bt=(0,0)+(0,0) = (0,0,0,0)."""
        a = SudokuMacroAdapter()
        addr = a.sparse_address(4, 4)
        assert addr == (0, 0, 0, 0)

    def test_address_encodes_bt_digits(self):
        """bt_d1 + bt_d2 is exactly what sparse_address returns."""
        a = SudokuMacroAdapter()
        hc = a.hypercell
        for r in range(9):
            for c in range(9):
                cell_dict = hc.cell(r, c)
                expected = cell_dict["bt_d1"] + cell_dict["bt_d2"]
                assert a.sparse_address(r, c) == expected


# ── Sudoku generator: core structural tests ───────────────────────────────────

class TestSudokuGenerator:
    def test_total_cells(self, frac_sudoku):
        assert len(frac_sudoku) == 729

    def test_reverse_index_size(self, frac_sudoku):
        assert len(frac_sudoku._reverse) == 729

    def test_unique_count(self, frac_sudoku):
        seen = set()
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        seen.add(frac_sudoku.sparse_address_6d(mr, mc, ur, uc))
        assert len(seen) == 729

    def test_all_coords_in_digit_set(self, frac_sudoku):
        for coords in frac_sudoku._reverse:
            assert all(c in (-1, 0, 1) for c in coords)

    def test_coords_length_is_6(self, frac_sudoku):
        assert len(frac_sudoku.sparse_address_6d(0, 0, 0, 0)) == 6

    def test_seam_verified(self, frac_sudoku):
        r = frac_sudoku.verify(silent=True)
        assert r["seam_verified"] is True

    def test_macro_oa_strength_4(self, frac_sudoku):
        """Sudoku macro addresses achieve OA strength 4."""
        r = frac_sudoku.verify(silent=True)
        assert r["macro_oa_strength"] == 4

    def test_round_trip(self, frac_sudoku):
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        coords   = frac_sudoku.sparse_address_6d(mr, mc, ur, uc)
                        pos_back = frac_sudoku._reverse[coords]
                        assert frac_sudoku.sparse_address_6d(*pos_back) == coords

    def test_cell_at_6d_returns_cell_pair(self, frac_sudoku):
        coords = frac_sudoku.sparse_address_6d(4, 4, 1, 1)
        pair   = frac_sudoku.cell_at_6d(coords)
        assert isinstance(pair, CellPair)

    def test_macro_is_cell_strata(self, frac_sudoku):
        coords = frac_sudoku.sparse_address_6d(0, 0, 0, 0)
        pair   = frac_sudoku.cell_at_6d(coords)
        assert isinstance(pair.macro, CellStrata)

    def test_micro_is_micro_cell(self, frac_sudoku):
        coords = frac_sudoku.sparse_address_6d(0, 0, 0, 0)
        pair   = frac_sudoku.cell_at_6d(coords)
        assert isinstance(pair.micro, MicroCell)

    def test_micro_suffix_matches(self, frac_sudoku):
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        coords = frac_sudoku.sparse_address_6d(mr, mc, ur, uc)
                        pair   = frac_sudoku.cell_at_6d(coords)
                        assert coords[4:] == pair.micro.coords_2d

    def test_macro_prefix_matches_adapter(self, frac_sudoku):
        """coords[:4] must equal the adapter's sparse_address for that macro cell."""
        adapter = SudokuMacroAdapter()
        for mr in range(9):
            for mc in range(9):
                macro_4d = adapter.sparse_address(mr, mc)
                for ur in range(3):
                    for uc in range(3):
                        coords = frac_sudoku.sparse_address_6d(mr, mc, ur, uc)
                        assert coords[:4] == macro_4d

    def test_centre_macro_address_is_zero(self, frac_sudoku):
        """Centre macro cell (4,4): bt(d1=5)+bt(d2=5) = (0,0,0,0)."""
        addr = frac_sudoku.sparse_address_6d(4, 4, 1, 1)
        assert addr[:4] == (0, 0, 0, 0)

    def test_invalid_address_raises(self, frac_sudoku):
        with pytest.raises(ValueError):
            frac_sudoku.cell_at_6d((2, 0, 0, 0, 0, 0))  # 2 never in {-1,0,1}

    def test_out_of_bounds_macro_raises(self, frac_sudoku):
        with pytest.raises(ValueError):
            frac_sudoku.sparse_address_6d(9, 0, 0, 0)

    def test_out_of_bounds_micro_raises(self, frac_sudoku):
        with pytest.raises(ValueError):
            frac_sudoku.sparse_address_6d(0, 0, 3, 0)

    def test_custom_micro_lo_shu(self):
        alt = np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]], dtype=np.int64)
        f = FractalHyperCell_3_6(generator="sudoku", micro_lo_shu=alt)
        assert len(f._reverse) == 729
        assert f.verify(silent=True)["seam_verified"]

    def test_bad_micro_shape_raises(self):
        with pytest.raises(ValueError):
            FractalHyperCell_3_6(generator="sudoku",
                                  micro_lo_shu=np.ones((4, 4), dtype=int))


# ── Product generator: same structural tests ──────────────────────────────────

class TestProductGenerator:
    def test_total_cells(self, frac_product):
        assert len(frac_product) == 729

    def test_seam_verified(self, frac_product):
        assert frac_product.verify(silent=True)["seam_verified"] is True

    def test_macro_oa_strength_4(self, frac_product):
        """FM-Dance also covers all of {-1,0,1}^4 (bijection onto torus)."""
        r = frac_product.verify(silent=True)
        assert r["macro_oa_strength"] == 4

    def test_generator_name(self, frac_product):
        assert frac_product.generator == "product"


# ── Generator comparison ──────────────────────────────────────────────────────

class TestGeneratorComparison:
    def test_addresses_differ_between_generators(self):
        """The two generators assign different 6D addresses to the same cells."""
        fs = FractalHyperCell_3_6.make_sudoku()
        fp = FractalHyperCell_3_6.make_product()
        differences = 0
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        s = fs.sparse_address_6d(mr, mc, ur, uc)
                        p = fp.sparse_address_6d(mr, mc, ur, uc)
                        if s != p:
                            differences += 1
        # Virtually all 729 addresses will differ — assert at least 500
        assert differences > 500, \
            f"Expected most addresses to differ, only {differences}/729 did"

    def test_both_cover_full_torus(self):
        """Both generators cover all of {-1,0,1}^6 exactly once."""
        import itertools
        all_6tuples = set(itertools.product((-1, 0, 1), repeat=6))
        for gen in ("sudoku", "product"):
            f = FractalHyperCell_3_6(generator=gen)
            addrs = {
                f.sparse_address_6d(mr, mc, ur, uc)
                for mr in range(9) for mc in range(9)
                for ur in range(3) for uc in range(3)
            }
            assert addrs == all_6tuples, \
                f"generator={gen!r} does not cover full {{-1,0,1}}^6"

    def test_sudoku_macro_is_bt_digit_structured(self):
        """
        Sudoku macro addresses directly encode BT digits of (d1, d2).
        Product macro addresses do NOT share this structure.
        """
        from flu.core.lo_shu_sudoku import LoShuSudokuHyperCell
        hc = LoShuSudokuHyperCell()
        fs = FractalHyperCell_3_6.make_sudoku()
        # For every macro cell, check bt encoding is transparent
        for r in range(9):
            for c in range(9):
                cell = hc.cell(r, c)
                expected_macro = cell["bt_d1"] + cell["bt_d2"]
                actual_macro   = fs.sparse_address_6d(r, c, 0, 0)[:4]
                assert actual_macro == expected_macro, \
                    f"Cell ({r},{c}): expected {expected_macro}, got {actual_macro}"


# ── Backward compat: explicit macro ──────────────────────────────────────────

class TestBackwardCompat:
    def test_explicit_macro_seam(self, frac_explicit):
        assert frac_explicit.verify(silent=True)["seam_verified"] is True

    def test_explicit_macro_is_product_generator(self, frac_explicit):
        assert frac_explicit.generator == "product"

    def test_explicit_macro_addresses_match_new_product(self, frac_explicit):
        """Explicit FLUHyperCell produces identical addresses to generator='product'."""
        fp = FractalHyperCell_3_6.make_product()
        for mr in range(9):
            for mc in range(9):
                for ur in range(3):
                    for uc in range(3):
                        assert (frac_explicit.sparse_address_6d(mr, mc, ur, uc) ==
                                fp.sparse_address_6d(mr, mc, ur, uc))

    def test_embed_as_3_6_is_product(self):
        """FLUHyperCell.embed_as_3_6() still uses product generator."""
        f = FLUHyperCell().embed_as_3_6()
        assert f.generator == "product"
        assert f.verify(silent=True)["seam_verified"]


# ── Repr and properties ───────────────────────────────────────────────────────

class TestReprAndProperties:
    def test_repr_contains_generator(self, frac_sudoku, frac_product):
        assert "sudoku" in repr(frac_sudoku)
        assert "product" in repr(frac_product)

    def test_len_both_generators(self, frac_sudoku, frac_product):
        assert len(frac_sudoku) == 729
        assert len(frac_product) == 729
