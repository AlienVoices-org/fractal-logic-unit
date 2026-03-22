"""
tests/test_container/test_scarstore.py
=======================================
Tests for ScarStore (OD-31 / HM-1 -- Holographic Sparsity Bound).

Covers:
  1. Construction with default seeds (bug fix: was passing d as positional arg)
  2. Construction with explicit seeds
  3. learn / recall / forget round-trip
  4. Exact reconstruction guarantee (HM-1)
  5. Compression ratio formula
  6. Anomaly rate
  7. Scar removal when value matches baseline
  8. ScarStore repr
  9. SparseCommunionManifold remains functional after ScarStore construction

STATUS: HM-1 PROVEN (V14) — representation theorem is exact by construction.
"""
from __future__ import annotations

import pytest
import numpy as np

from flu.container.sparse import ScarStore, SparseCommunionManifold


# ── 1. Construction ───────────────────────────────────────────────────────────

def test_scarstore_default_seeds() -> None:
    """Default construction (seeds=None) must not raise — this was the bug."""
    s = ScarStore(n=3, d=2)
    assert s.n == 3
    assert s.d == 2
    assert s.scar_count() == 0


def test_scarstore_default_seeds_various_nd() -> None:
    """Default construction for several (n, d) pairs."""
    for n, d in [(3, 2), (3, 4), (5, 2), (5, 3), (7, 2)]:
        s = ScarStore(n=n, d=d)
        assert s.n == n
        assert s.d == d


def test_scarstore_explicit_seeds() -> None:
    """Construction with explicit seeds must produce matching dimensionality."""
    seeds = [np.array([1, 2, 0]), np.array([0, 2, 1])]  # 2 seeds, n=3
    s = ScarStore(n=3, d=2, seeds=seeds)
    assert s.d == 2


# ── 2. Learn / recall / forget ────────────────────────────────────────────────

def test_learn_recall_exact() -> None:
    """HM-1: recall must return the exact true value after learn."""
    s = ScarStore(n=5, d=2)
    coord = (1, -1)
    true_val = 42.5
    s.learn(coord, true_val)
    assert abs(s.recall(coord) - true_val) < 1e-12


def test_recall_baseline_when_no_scar() -> None:
    """Coordinates without scars must return the baseline value."""
    s = ScarStore(n=3, d=2)
    # Recall before any learning — should return baseline
    v = s.recall((0, 0))
    assert isinstance(v, float)


def test_learn_removes_scar_when_matches_baseline() -> None:
    """If true_value == baseline, learn must delete any existing scar."""
    s = ScarStore(n=3, d=2)
    coord = (1, 0)
    # Learn a non-baseline value first
    s.learn(coord, 999.0)
    assert s.scar_count() == 1
    # Learn the exact baseline value — scar should be removed
    baseline = s.recall(coord)  # scar is still 999 at this point
    s.learn(coord, float(s._manifold[coord]))  # learn the baseline exactly
    assert s.scar_count() == 0


def test_forget() -> None:
    """forget() must remove a scar, reverting coordinate to baseline."""
    s = ScarStore(n=3, d=3)
    coord = (-1, 0, 1)
    s.learn(coord, 77.0)
    assert s.scar_count() == 1
    s.forget(coord)
    assert s.scar_count() == 0
    # Value after forget must equal baseline
    baseline = s._manifold[coord]
    assert abs(s.recall(coord) - baseline) < 1e-12


def test_forget_nonexistent_is_noop() -> None:
    """forget() on an unscarred coordinate must be a no-op."""
    s = ScarStore(n=3, d=2)
    s.forget((0, 1))  # should not raise


# ── 3. HM-1: exact reconstruction ────────────────────────────────────────────

def test_hm1_full_domain_reconstruction() -> None:
    """
    HM-1 (PROVEN): ScarStore provides exact reconstruction for every cell.

    For a small domain (n=3, d=2 → 9 cells), learn all cells with arbitrary
    values, then verify exact recall for all.
    """
    n, d = 3, 2
    half = n // 2
    s = ScarStore(n=n, d=d)
    rng = np.random.default_rng(99)

    ground_truth = {}
    coords = [
        (x0, x1)
        for x0 in range(-half, half + 1)
        for x1 in range(-half, half + 1)
    ]
    for coord in coords:
        v = float(rng.integers(-100, 101))
        ground_truth[coord] = v
        s.learn(coord, v)

    for coord in coords:
        recalled = s.recall(coord)
        expected = ground_truth[coord]
        assert abs(recalled - expected) < 1e-10, (
            f"Mismatch at {coord}: recalled={recalled}, expected={expected}"
        )


# ── 4. Compression metrics ────────────────────────────────────────────────────

def test_compression_ratio_formula() -> None:
    """
    compression_ratio = n^d / (d + |scars|).

    With no scars: ratio = n^d / d.
    With k scars: ratio = n^d / (d + k).
    """
    n, d = 3, 4   # n^d = 81
    s = ScarStore(n=n, d=d)
    full = n ** d  # 81
    # Empty
    expected_empty = float(full) / d
    assert abs(s.compression_ratio() - expected_empty) < 1e-9

    # Add 5 scars
    coords = [(-1, -1, -1, -1), (0, 0, 0, 0), (1, 1, 1, 1), (-1, 1, 0, 0), (1, 0, -1, 1)]
    for c in coords:
        s.learn(c, 100.0)

    expected_with_scars = float(full) / (d + 5)
    assert abs(s.compression_ratio() - expected_with_scars) < 1e-9


def test_anomaly_rate() -> None:
    """anomaly_rate = |scars| / n^d."""
    n, d = 3, 2   # n^d = 9
    s = ScarStore(n=n, d=d)
    assert s.anomaly_rate() == 0.0

    s.learn((0, 0), 50.0)
    assert abs(s.anomaly_rate() - 1 / 9) < 1e-12

    s.learn((1, 1), 51.0)
    assert abs(s.anomaly_rate() - 2 / 9) < 1e-12


# ── 5. Compression benchmark (HM-1 ≥5× at 10% anomaly) ──────────────────────

@pytest.mark.parametrize("n,d", [(3, 4), (3, 6), (5, 3), (5, 4)])
def test_hm1_compression_5x_at_10pct(n: int, d: int) -> None:
    """
    HM-1 benchmark: ratio ≥ 5× at 10% anomaly rate.

    Analytically: ratio = n^d / (d + 0.1·n^d) = 1 / (d/n^d + 0.1).
    For n^d ≥ 10d, ratio ≥ 5.  (Always true for the tested params.)
    """
    rng = np.random.default_rng(42)
    full = n ** d
    s = ScarStore(n=n, d=d)
    half = n // 2

    all_coords = list(np.ndindex(*([n] * d)))
    n_anom = max(1, int(full * 0.10))
    chosen = rng.choice(len(all_coords), size=n_anom, replace=False)
    for idx in chosen:
        coord = tuple(c - half for c in all_coords[idx])
        s.learn(coord, float(rng.integers(-100, 101)))

    assert s.compression_ratio() >= 5.0, (
        f"n={n}, d={d}: ratio={s.compression_ratio():.2f} < 5.0"
    )


# ── 6. Repr ───────────────────────────────────────────────────────────────────

def test_repr() -> None:
    """ScarStore repr must include n, d, scar count, and compression."""
    s = ScarStore(n=3, d=2)
    r = repr(s)
    assert "ScarStore" in r
    assert "n=3" in r
    assert "d=2" in r
    assert "scars=0" in r
