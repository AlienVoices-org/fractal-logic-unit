"""
tests/test_core/test_fractal_net_orthogonal.py
===============================================
Tests for FractalNetOrthogonal — the DN1 OA(n⁴,4,n,4) digital net.

Covers:
  - Construction for n ∈ {3,5,7}
  - OA(n⁴,4,n,4) verification (all n⁴ 4-tuples unique)
  - generate() shape, range, dtype
  - generate_scrambled() — preserves OA structure, different from plain
  - Prefix property: first n² points cover {0,...,n-1}^4 uniformly per row
  - Comparison with FractalNet (same point SET at full N, different order)
  - Error handling (even n, n<3)
  - Package import
  - Benchmarks: L2* at partial N confirms prefix advantage

Theorem references:
  DN1, DN1-GL, DN1-OA, DN1-GEN — all PROVEN V15.3.2
"""

import itertools
import pytest
import numpy as np

from flu.core.fractal_net import FractalNet, FractalNetOrthogonal
from flu.core.fm_dance import index_to_coords


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", params=[3, 5, 7])
def net(request):
    return FractalNetOrthogonal(n=request.param)

@pytest.fixture(scope="module")
def net3():
    return FractalNetOrthogonal(n=3)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_dimension_is_always_4(self, net):
        assert net.d == 4

    def test_base_volume_is_n4(self, net):
        assert net.N == net.n ** 4

    def test_base_block_shape(self, net):
        assert net._base_block.shape == (net.n ** 4, 4)

    def test_base_block_dtype(self, net):
        assert net._base_block.dtype == np.float64

    def test_base_block_values_in_range(self, net):
        b = net._base_block.astype(int)
        assert b.min() == 0
        assert b.max() == net.n - 1

    def test_repr_contains_oa(self, net):
        assert "OA=True" in repr(net)
        assert str(net.n) in repr(net)

    def test_odd_n_only(self):
        with pytest.raises(ValueError, match="odd"):
            FractalNetOrthogonal(n=4)

    def test_n_lt_3_raises(self):
        with pytest.raises(ValueError):
            FractalNetOrthogonal(n=1)


# ── OA(n⁴,4,n,4) certificate ─────────────────────────────────────────────────

class TestOACertificate:
    def test_verify_oa_passes(self, net):
        """verify_oa() must report all_pass=True and strength=4."""
        result = net.verify_oa()
        assert result["all_pass"], f"n={net.n}: {result}"
        assert result["oa_strength"] == 4
        assert result["unique_tuples"] == net.n ** 4

    def test_all_n4_tuples_unique(self, net):
        """All n⁴ rows of the base block are distinct n-ary 4-tuples."""
        digits = net._base_block.astype(int)
        tuples = set(map(tuple, digits))
        assert len(tuples) == net.n ** 4

    def test_covers_full_alphabet(self, net):
        """All n-ary 4-tuples in {0,...,n-1}^4 appear (OA strength 4)."""
        digits = net._base_block.astype(int)
        actual = set(map(tuple, digits))
        expected = set(itertools.product(range(net.n), repeat=4))
        assert actual == expected

    def test_2d_marginals_uniform(self, net):
        """Each 2D marginal has exactly n² unique pairs (OA strength ≥ 2)."""
        digits = net._base_block.astype(int)
        for i, j in itertools.combinations(range(4), 2):
            proj = set(zip(digits[:, i], digits[:, j]))
            assert len(proj) == net.n ** 2, \
                f"n={net.n}, dims ({i},{j}): {len(proj)} ≠ {net.n**2}"

    def test_same_point_set_as_fractal_net(self, net3):
        """FractalNetOrthogonal and FractalNet cover the same n⁴ points (DN1-OA)."""
        N = net3.n ** 4
        oa_pts = net3.generate(N)
        fn_pts = FractalNet(net3.n, 4).generate(N)

        oa_set = set(map(tuple, np.round(oa_pts, 9)))
        fn_set = set(map(tuple, np.round(fn_pts, 9)))
        assert oa_set == fn_set, \
            "OA net and FractalNet should cover the same n⁴ lattice points"


# ── generate() ────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_shape(self, net):
        N = net.n ** 4
        pts = net.generate(N)
        assert pts.shape == (N, 4)

    def test_range(self, net):
        pts = net.generate(net.n ** 4)
        assert float(pts.min()) >= 0.0
        assert float(pts.max()) < 1.0

    def test_zero_points(self, net):
        pts = net.generate(0)
        assert pts.shape == (0, 4)

    def test_partial_n_then_full(self, net3):
        """Points at full N are always the same regardless of num_points."""
        N = net3.n ** 4
        pts_full = net3.generate(N)
        pts_partial = net3.generate(N // 3)
        # Partial prefix must match beginning of full sequence
        np.testing.assert_array_equal(
            pts_partial, pts_full[:N//3],
            err_msg="Partial generate must be a prefix of full generate"
        )

    def test_beyond_base_block(self, net3):
        """Multi-depth generation: N > n⁴ should work."""
        N = net3.n ** 4 * 2
        pts = net3.generate(N)
        assert pts.shape == (N, 4)
        assert float(pts.min()) >= 0.0
        assert float(pts.max()) < 1.0


# ── Prefix property ───────────────────────────────────────────────────────────

class TestPrefixProperty:
    def test_first_n2_points_cover_all_digits(self, net):
        """
        The first n² points form a Latin 'row' of the Sudoku grid.
        In digit space they should cover all n-ary values in at least
        some dimensions — not all (it's a row not the full block).
        At minimum the first n² points should be well-spread.
        """
        n = net.n
        pts = net.generate(n ** 2)
        assert pts.shape == (n ** 2, 4)
        # Each row of the Sudoku grid has distinct d1 values →
        # the first n² points have distinct d1 encodings
        digits = (pts * n + 0.5).astype(int)
        # First two coordinates encode pos(d1) — should vary across n² points
        coord_pairs = set(zip(digits[:, 0], digits[:, 1]))
        assert len(coord_pairs) == n ** 2, \
            f"First n²={n**2} points should have distinct (d1-row, d1-col) pairs"

    def test_prefix_l2star_better_than_fractalnet(self, net3):
        """
        DN1 prefix property: at N=n², OA ordering achieves much better
        L2* than FractalNet (10× better at n=3 — the key benchmark result).
        """
        n = net3.n
        N = n ** 2  # first 9 points for n=3

        oa_pts = net3.generate(N)
        fn_pts = FractalNet(n, 4).generate(N)

        def warnock(pts):
            Np, d = pts.shape
            s1 = float(np.sum(np.prod(1 - pts**2 / 2, axis=1)))
            mx = np.maximum(pts[:, None, :], pts[None, :, :])
            s2 = float(np.sum(np.prod(1 - mx, axis=2)))
            return float(np.sqrt(abs(3**(-d) - 2**(1-d)/Np*s1 + s2/Np**2)))

        oa_disc = warnock(oa_pts)
        fn_disc = warnock(fn_pts)

        assert oa_disc < fn_disc, (
            f"OA ordering should have better L2* at N={N}: "
            f"OA={oa_disc:.4f} vs FN={fn_disc:.4f}"
        )
        # For n=3 the advantage is dramatic (>5× at N=n²=9)
        if n == 3:
            assert oa_disc * 5 < fn_disc, \
                f"n=3 advantage at N=9 should be >5×: OA={oa_disc:.4f}, FN={fn_disc:.4f}"


# ── generate_scrambled() ──────────────────────────────────────────────────────

class TestGenerateScrambled:
    def test_shape(self, net):
        pts = net.generate_scrambled(net.n ** 4)
        assert pts.shape == (net.n ** 4, 4)

    def test_range(self, net):
        pts = net.generate_scrambled(net.n ** 4)
        assert float(pts.min()) >= 0.0
        assert float(pts.max()) < 1.0

    def test_different_from_plain(self, net3):
        """Scrambled should differ from plain (with overwhelming probability)."""
        N = net3.n ** 4
        plain = net3.generate(N)
        scr   = net3.generate_scrambled(N)
        assert not np.allclose(plain, scr), \
            "Scrambled should differ from plain ordering"

    def test_different_seeds_differ(self, net3):
        """Different seed_rank should give different scrambled sequences."""
        N = net3.n ** 4
        s0 = net3.generate_scrambled(N, seed_rank=0)
        s1 = net3.generate_scrambled(N, seed_rank=1)
        assert not np.allclose(s0, s1)

    def test_scrambled_preserves_oa(self, net3):
        """
        APN scrambling preserves OA(n⁴,4,n,4): each scrambled depth block
        still covers all n-ary 4-tuples exactly once (bijection per axis).
        """
        n = net3.n
        N = n ** 4
        pts = net3.generate_scrambled(N)
        # Map back to digits
        digits = (pts * n + 0.5).astype(int)
        tuples = set(map(tuple, digits))
        assert len(tuples) == N, \
            f"Scrambled OA should have {N} unique 4-tuples, got {len(tuples)}"

    def test_scrambled_is_different_oa_instance(self, net3):
        """
        FLU-Owen scrambling applies independent APN permutations per column.
        The scrambled result is a DIFFERENT OA(n⁴,4,n,4) instance — it covers
        a different set of 81 4-tuples, but still OA strength 4. This is the
        correct Owen behaviour: the scrambled net is a random rotation of the
        original, not the same point set in a different order.
        """
        N = net3.n ** 4
        plain = net3.generate(N)
        scr   = net3.generate_scrambled(N)
        # Sets should differ (independent per-column permutations)
        plain_set = set(map(tuple, np.round(plain, 9)))
        scr_set   = set(map(tuple, np.round(scr, 9)))
        assert plain_set != scr_set, \
            "Scrambled should be a different OA instance than plain"
        # But scrambled must still be OA(n⁴,4,n,4)
        n = net3.n
        scr_digits = (scr * n + 0.5).astype(int)
        tuples = set(map(tuple, scr_digits))
        assert len(tuples) == N, \
            f"Scrambled must still have {N} unique 4-tuples (OA preserved)"

    def test_owen_mode_works(self, net3):
        pts = net3.generate_scrambled(net3.n**4, mode="owen")
        assert pts.shape == (net3.n**4, 4)

    def test_coordinated_mode_works(self, net3):
        pts = net3.generate_scrambled(net3.n**4, mode="coordinated")
        assert pts.shape == (net3.n**4, 4)


# ── Package export ─────────────────────────────────────────────────────────────

class TestPackageExport:
    def test_importable_from_flu(self):
        import flu
        assert hasattr(flu, "FractalNetOrthogonal")

    def test_importable_from_core(self):
        from flu.core.fractal_net import FractalNetOrthogonal
        assert FractalNetOrthogonal is not None

    def test_smoke_via_flu_namespace(self):
        import flu
        net = flu.FractalNetOrthogonal(n=3)
        pts = net.generate(81)
        assert pts.shape == (81, 4)


# ── Theorem registry ──────────────────────────────────────────────────────────

class TestTheoremConnections:
    def test_dn1_gen_proven_all_odd_n(self):
        """DN1-GEN PROVEN: OA(n⁴,4,n,4) for n=3,5,7 by det=4 argument."""
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DN1-GEN")
        assert t is not None and t.status == "PROVEN"
        # Computational certificate for each n
        for n in [3, 5, 7]:
            net = FractalNetOrthogonal(n=n)
            r = net.verify_oa()
            assert r["all_pass"], f"DN1-GEN computational cert failed for n={n}"

    def test_dn1_oa_proven(self):
        """DN1-OA PROVEN: OA(n⁴,4,n,4) for n=3 with strength=4."""
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DN1-OA")
        assert t is not None and t.status == "PROVEN"

    def test_dn1_proven(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DN1")
        assert t is not None and t.status == "PROVEN"
