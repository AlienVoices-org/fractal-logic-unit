"""
tests/test_core/test_lo_shu_sudoku.py
===========================
Tests for the Lo Shu Sudoku 3^4 Hypercell and DN1 partial resolution.
All tests run without pytest; compatible with run_tests.py.
"""

from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flu.core.lo_shu_sudoku import (
    LoShuSudokuHyperCell,
    LO_SHU,
    generate_d1,
    generate_d2,
    verify_digital_net_property,
    make_hypercell,
    _to_bt2,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ── Generation tests ──────────────────────────────────────────────────────────

def test_d1_is_valid_sudoku() -> None:
    """d1 layer must be a valid 9×9 Sudoku."""
    d1 = generate_d1()
    for r in range(9):
        _assert(sorted(d1[r, :]) == list(range(1, 10)), f"d1 row {r} not a permutation")
    for c in range(9):
        _assert(sorted(d1[:, c]) == list(range(1, 10)), f"d1 col {c} not a permutation")
    for br in range(3):
        for bc in range(3):
            blk = sorted(d1[3*br:3*br+3, 3*bc:3*bc+3].flatten())
            _assert(blk == list(range(1, 10)), f"d1 block ({br},{bc}) not a permutation")


def test_d2_is_valid_sudoku() -> None:
    """d2 layer must be a valid 9×9 Sudoku."""
    d2 = generate_d2()
    for r in range(9):
        _assert(sorted(d2[r, :]) == list(range(1, 10)), f"d2 row {r} not a permutation")
    for c in range(9):
        _assert(sorted(d2[:, c]) == list(range(1, 10)), f"d2 col {c} not a permutation")
    for br in range(3):
        for bc in range(3):
            blk = sorted(d2[3*br:3*br+3, 3*bc:3*bc+3].flatten())
            _assert(blk == list(range(1, 10)), f"d2 block ({br},{bc}) not a permutation")


def test_graeco_latin() -> None:
    """All 81 (d1, d2) pairs must be unique — Graeco-Latin square property."""
    d1, d2 = generate_d1(), generate_d2()
    pairs = set(zip(d1.flatten().tolist(), d2.flatten().tolist()))
    _assert(len(pairs) == 81, f"Only {len(pairs)} unique (d1,d2) pairs, expected 81")


def test_norm1_coverage() -> None:
    """norm1 = (d1-1)*9 + d2 must cover 1..81 exactly once."""
    cell = LoShuSudokuHyperCell()
    vals = sorted(cell.norm1.flatten().tolist())
    _assert(vals == list(range(1, 82)), "norm1 does not cover 1..81 exactly")


def test_balanced_sum_zero() -> None:
    """Sum of balanced values must be 0."""
    cell = LoShuSudokuHyperCell()
    total = int(np.sum(cell.balanced))
    _assert(total == 0, f"Balanced sum = {total}, expected 0")


def test_centre_cell_is_zero() -> None:
    """Centre cell (4,4) must have d1=5, d2=5, balanced=0."""
    cell = LoShuSudokuHyperCell()
    c = cell.cell(4, 4)
    _assert(c["d1"] == 5, f"Centre d1 = {c['d1']}, expected 5")
    _assert(c["d2"] == 5, f"Centre d2 = {c['d2']}, expected 5")
    _assert(c["balanced"] == 0, f"Centre balanced = {c['balanced']}, expected 0")
    _assert(c["bt_d1"] == (0, 0), f"Centre bt_d1 = {c['bt_d1']}, expected (0,0)")


def test_unity_sums_to_one() -> None:
    """Unity field must sum to 1.0."""
    cell = LoShuSudokuHyperCell()
    total = float(np.sum(cell.unity))
    _assert(abs(total - 1.0) < 1e-10, f"Unity sum = {total}, expected 1.0")


def test_centre_block_is_lo_shu() -> None:
    """The centre 3×3 block of d1 must equal the Lo Shu (block of d1)."""
    d1 = generate_d1()
    centre = d1[3:6, 3:6]
    _assert(np.array_equal(centre, LO_SHU), f"Centre block d1 ≠ Lo Shu")


# ── Digital net / OA tests ────────────────────────────────────────────────────

def test_fractal_net_points_shape() -> None:
    """to_fractal_net_points() must return shape (81, 4)."""
    cell = LoShuSudokuHyperCell()
    pts = cell.to_fractal_net_points()
    _assert(pts.shape == (81, 4), f"Points shape = {pts.shape}, expected (81,4)")


def test_fractal_net_points_on_lattice() -> None:
    """All points must lie on the {0, 1/3, 2/3}^4 lattice."""
    cell = LoShuSudokuHyperCell()
    pts = cell.to_fractal_net_points()
    valid = set([0.0, 1/3, 2/3])
    for d in range(4):
        for v in pts[:, d]:
            _assert(any(abs(v - x) < 1e-9 for x in valid),
                    f"Point {v} not on {{0, 1/3, 2/3}} lattice in dim {d}")


def test_oa_strength_4() -> None:
    """
    OA(81,4,3,4): every 4-tuple from {0,1,2}^4 appears exactly once.
    This is the strongest possible OA for 81 = 3^4 runs.
    """
    cell = LoShuSudokuHyperCell()
    pts = cell.to_fractal_net_points()
    tuples = set(tuple((pts[k] * 3 + 0.5).astype(int).tolist()) for k in range(81))
    _assert(len(tuples) == 81, f"Only {len(tuples)} unique 4-tuples, expected 81")


def test_oa_2d_marginals() -> None:
    """
    Every 2D marginal must have exactly 9 points per (a/3, b/3) cell.
    This is OA strength ≥ 2.
    """
    from itertools import combinations
    cell = LoShuSudokuHyperCell()
    pts = cell.to_fractal_net_points()
    for i, j in combinations(range(4), 2):
        for a in range(3):
            for b in range(3):
                cnt = int(np.sum(
                    (np.abs(pts[:, i] - a/3) < 1e-9) &
                    (np.abs(pts[:, j] - b/3) < 1e-9)
                ))
                _assert(cnt == 9, f"Dims ({i},{j}) cell ({a},{b}) has {cnt} pts, expected 9")


def test_finest_grain_net_0_4_4() -> None:
    """
    At finest resolution 1/3^4, each cell has exactly 1 point.
    This is the (0,4,4)-net property at the natural lattice scale.
    """
    cell = LoShuSudokuHyperCell()
    pts = cell.to_fractal_net_points()
    for a in range(3):
        for b in range(3):
            for c_ in range(3):
                for d in range(3):
                    cnt = int(np.sum(
                        (np.abs(pts[:, 0] - a/3) < 1e-9) &
                        (np.abs(pts[:, 1] - b/3) < 1e-9) &
                        (np.abs(pts[:, 2] - c_/3) < 1e-9) &
                        (np.abs(pts[:, 3] - d/3) < 1e-9)
                    ))
                    _assert(cnt == 1, f"Cell ({a},{b},{c_},{d}) has {cnt} pts, expected 1")


def test_full_net_classification_t3() -> None:
    """Full (t,4,4)-net parameter must be t=3 (not better, not worse)."""
    result = verify_digital_net_property(verbose=False)
    _assert(result["net_t_full"] == 3,
            f"Full net t = {result['net_t_full']}, expected 3")


def test_dn1_certificate_passes() -> None:
    """Combined DN1 certificate check."""
    result = verify_digital_net_property(verbose=False)
    _assert(result["all_pass"], f"DN1 certificate failed: {result}")


def test_bt2_mapping() -> None:
    """Balanced ternary 2-trit mapping covers -4..4 without collision."""
    seen = set()
    for v in range(1, 10):
        bt = _to_bt2(v)
        _assert(bt not in seen, f"BT collision at v={v}: {bt}")
        seen.add(bt)
        t1, t0 = bt
        _assert(t1 in (-1, 0, 1), f"t1={t1} out of range")
        _assert(t0 in (-1, 0, 1), f"t0={t0} out of range")
        val_check = t1 * 3 + t0
        _assert(val_check == v - 5, f"BT decode error: {bt} → {val_check} ≠ {v-5}")


def test_make_hypercell_convenience() -> None:
    """make_hypercell() must return an equivalent cell."""
    cell = make_hypercell()
    result = cell.verify()
    _assert(result["all_pass"], "make_hypercell() verification failed")


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_d1_is_valid_sudoku,
    test_d2_is_valid_sudoku,
    test_graeco_latin,
    test_norm1_coverage,
    test_balanced_sum_zero,
    test_centre_cell_is_zero,
    test_unity_sums_to_one,
    test_centre_block_is_lo_shu,
    test_fractal_net_points_shape,
    test_fractal_net_points_on_lattice,
    test_oa_strength_4,
    test_oa_2d_marginals,
    test_finest_grain_net_0_4_4,
    test_full_net_classification_t3,
    test_dn1_certificate_passes,
    test_bt2_mapping,
    test_make_hypercell_convenience,
]


if __name__ == "__main__":
    passed = failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  ✓  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {fn.__name__}: {e}")
            failed += 1
    print(f"\n  PASSED {passed}   FAILED {failed}   TOTAL {passed+failed}")
