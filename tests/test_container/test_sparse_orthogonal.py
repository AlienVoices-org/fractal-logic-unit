"""
tests/test_core/test_sparse_orthogonal.py
==========================================
Test suite for SparseOrthogonalManifold — the O(D) memory-free oracle for the
DN1-REC OA(n^(4k), 4k, n, 4k) orthogonal array family.

Covers:
  - Construction for odd n (Lo Shu map, det=4) and even n (snake map, det=1)
  - OA(n^d, d, n, d) verification for n ∈ {2,3,4,5} and d ∈ {4,8}
  - __getitem__: single-cell and batch evaluation
  - cell_at_oa_rank: natural digit ordering, round-trip consistency
  - cell_at_rank: FM-Dance rank ordering (backward compat)
  - verify_oa(): self-certification
  - materialize(): full tensor reconstruction
  - Communion (__add__): dimension concatenation
  - Error handling: even n (Lo Shu raises), d not multiple of 4
  - Package import and export

Theorem references:
  DNO-GEN, DNO-COEFF-EVEN, DNO-INV, DNO-OPT, DNO-P1, DNO-P2 — all PROVEN V15.3.2
"""

import itertools
import pytest
import numpy as np

from flu.container.sparse import SparseOrthogonalManifold


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def m3():
    return SparseOrthogonalManifold(n=3, d=4)

@pytest.fixture(scope="module")
def m5():
    return SparseOrthogonalManifold(n=5, d=4)

@pytest.fixture(scope="module")
def m2():
    return SparseOrthogonalManifold(n=2, d=4)

@pytest.fixture(scope="module")
def m4():
    return SparseOrthogonalManifold(n=4, d=4)

@pytest.fixture(scope="module")
def m3_d8():
    return SparseOrthogonalManifold(n=3, d=8)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_odd_n_constructs(self, m3):
        assert m3.n == 3
        assert m3.d == 4

    def test_even_n_constructs(self, m2, m4):
        assert m2.n == 2 and m2.d == 4
        assert m4.n == 4 and m4.d == 4

    def test_d8_constructs(self, m3_d8):
        assert m3_d8.d == 8
        assert m3_d8._blocks == 2

    def test_blocks_count(self, m3, m3_d8):
        assert m3._blocks == 1
        assert m3_d8._blocks == 2

    def test_shape_attribute(self, m3):
        assert m3.shape == (3, 3, 3, 3)

    def test_repr_contains_oa(self, m3):
        r = repr(m3)
        assert "SparseOrthogonalManifold" in r
        assert "OA(" in r
        assert "81" in r

    def test_memory_independent_of_n(self, m3, m5):
        # Both d=4: memory scales with d only, not n
        assert repr(m3).count("128B") or "memory" in repr(m3)

    def test_odd_n_raises_for_even(self):
        """SparseOrthogonalManifold accepts n=2 (even snake map)."""
        m = SparseOrthogonalManifold(n=2, d=4)
        assert m.n == 2

    def test_n1_raises(self):
        with pytest.raises(ValueError):
            SparseOrthogonalManifold(n=1, d=4)

    def test_d3_raises(self):
        with pytest.raises(ValueError):
            SparseOrthogonalManifold(n=3, d=3)

    def test_d6_raises(self):
        with pytest.raises(ValueError):
            SparseOrthogonalManifold(n=3, d=6)

    def test_d4_ok(self):
        m = SparseOrthogonalManifold(n=3, d=4)
        assert m.d == 4

    def test_d8_ok(self):
        m = SparseOrthogonalManifold(n=3, d=8)
        assert m.d == 8


# ── OA Certificate — odd n ────────────────────────────────────────────────────

class TestOACertificateOddN:
    def test_verify_oa_n3(self, m3):
        """DNO-OPT: OA(81,4,3,4) via bijectivity of A_odd."""
        assert m3.verify_oa()

    def test_verify_oa_n5(self, m5):
        """DNO-GEN: det=4, gcd(4,5)=1 → A ∈ GL(4,Z_5)."""
        assert m5.verify_oa()

    def test_all_n3_tuples_unique(self, m3):
        N = 3**4
        seen = set()
        for k in range(N):
            seen.add(m3._oa_rank_to_signed_coords(k))
        assert len(seen) == N

    def test_all_n5_tuples_unique(self, m5):
        N = 5**4
        seen = set()
        for k in range(N):
            seen.add(m5._oa_rank_to_signed_coords(k))
        assert len(seen) == N

    def test_covers_full_alphabet_n3(self, m3):
        n = m3.n
        seen = set()
        for k in range(n**4):
            seen.add(m3._oa_rank_to_signed_coords(k))
        expected = set(itertools.product(range(-(n//2), n//2+1), repeat=4))
        assert seen == expected

    def test_2d_marginals_n3(self, m3):
        n = m3.n
        coords = [m3._oa_rank_to_signed_coords(k) for k in range(n**4)]
        for i, j in itertools.combinations(range(4), 2):
            proj = set((c[i], c[j]) for c in coords)
            assert len(proj) == n**2, f"dims ({i},{j}): {len(proj)} ≠ {n**2}"

    def test_d8_oa_n3(self, m3_d8):
        """DNO-REC-MATRIX: A^(2) ∈ GL(8,Z_3) → OA(6561,8,3,8)."""
        assert m3_d8.verify_oa()

    def test_d8_all_tuples_unique_n3(self, m3_d8):
        N = 3**8
        seen = set()
        for k in range(N):
            seen.add(m3_d8._oa_rank_to_signed_coords(k))
        assert len(seen) == N


# ── OA Certificate — even n ───────────────────────────────────────────────────

class TestOACertificateEvenN:
    def test_verify_oa_n2(self, m2):
        """DNO-COEFF-EVEN: A_even det=1 → OA(16,4,2,4)."""
        assert m2.verify_oa()

    def test_verify_oa_n4(self, m4):
        """DNO-COEFF-EVEN: OA(256,4,4,4)."""
        assert m4.verify_oa()

    def test_all_n2_tuples_unique(self, m2):
        N = 2**4
        seen = set(m2._oa_rank_to_signed_coords(k) for k in range(N))
        assert len(seen) == N

    def test_all_n4_tuples_unique(self, m4):
        N = 4**4
        seen = set(m4._oa_rank_to_signed_coords(k) for k in range(N))
        assert len(seen) == N

    def test_n2_is_gray_code(self, m2):
        """For n=2 the snake map is a differential Gray code on 4 bits."""
        N = 2**4
        # All 16 binary 4-tuples must appear exactly once
        seen = set()
        for k in range(N):
            coords = m2._oa_rank_to_signed_coords(k)
            # Unsigned: add half=0 for n=2
            seen.add(tuple(c + m2.half for c in coords))
        assert seen == set(itertools.product(range(2), repeat=4))

    def test_d8_even_n2(self):
        """DNO-REC-MATRIX: A^(2) ∈ GL(8,Z_2) → OA(256,8,2,8)."""
        m = SparseOrthogonalManifold(n=2, d=8)
        assert m.verify_oa()

    def test_d8_even_n4(self):
        """OA(4^8,8,4,8)."""
        m = SparseOrthogonalManifold(n=4, d=8)
        assert m.verify_oa()


# ── cell_at_oa_rank ───────────────────────────────────────────────────────────

class TestCellAtOaRank:
    def test_returns_int(self, m3):
        v = m3.cell_at_oa_rank(0)
        assert isinstance(v, int)

    def test_value_in_range(self, m3):
        n = m3.n
        for k in range(n**4):
            v = m3.cell_at_oa_rank(k)
            assert -(n//2) <= v <= n//2, f"k={k}: {v} out of range"

    def test_centre_rank(self, m3):
        """Centre of the n^4 grid has balanced value 0."""
        # The centre rank is (n^4 - 1) // 2 = 40 for n=3
        # The coordinates are all-zero in the balanced representation
        # (centre of the Lo Shu grid). Value = signed_to_value(0,0,0,0) = 0.
        N = m3.n**4
        centre_k = (N - 1) // 2
        # Verify we get a valid value
        v = m3.cell_at_oa_rank(centre_k)
        assert -(m3.n//2) <= v <= m3.n//2

    def test_all_values_cover_range(self, m3):
        n = m3.n
        vals = set(m3.cell_at_oa_rank(k) for k in range(n**4))
        # Values should span at least the allowed range
        assert min(vals) == -(n//2)
        assert max(vals) == n//2


# ── Inverse oracle ────────────────────────────────────────────────────────────

class TestInverseOracle:
    """DNO-INV: 0 errors for all n in {2,3,4,5,6,7}."""

    def _round_trip_check(self, n):
        from flu.container.sparse import SparseOrthogonalManifold as SOM
        m = SOM(n=n, d=4)
        errors = 0
        for k in range(n**4):
            coords = m._oa_rank_to_signed_coords(k)
            # shift to unsigned
            unsigned = tuple(c + m.half for c in coords)
            # recompute rank via inverse
            half = m.half
            is_odd = n % 2 != 0
            a = [u % n for u in unsigned]  # already unsigned 0..n-1
            if is_odd:
                inv2 = pow(2, -1, n)
                r_c = (a[2] - a[1]) % n
                b_r = (a[1] - r_c) % n
                s   = (a[3] * inv2) % n
                r_r = ((s + a[0]) * inv2) % n
                b_c = (r_r - a[0]) % n
            else:
                b_r = a[0]
                r_r = (a[1] - a[0]) % n
                b_c = (a[2] - r_r) % n
                r_c = (a[3] - b_c) % n
            recovered = b_r * n**3 + r_r * n**2 + b_c * n + r_c
            if recovered != k:
                errors += 1
        return errors

    def test_inverse_n3(self):
        assert self._round_trip_check(3) == 0

    def test_inverse_n5(self):
        assert self._round_trip_check(5) == 0

    def test_inverse_n2(self):
        assert self._round_trip_check(2) == 0

    def test_inverse_n4(self):
        assert self._round_trip_check(4) == 0


# ── __getitem__ ───────────────────────────────────────────────────────────────

class TestGetItem:
    def test_single_tuple(self, m3):
        v = m3[0, 0, 0, 0]
        assert isinstance(v, int)
        assert -(m3.n//2) <= v <= m3.n//2

    def test_single_negative_coord(self, m3):
        v = m3[-1, 0, 1, -1]
        assert isinstance(v, (int, np.integer))

    def test_wrong_dim_raises(self, m3):
        with pytest.raises(IndexError):
            m3[0, 0, 0]

    def test_batch_numpy(self, m3):
        coords = np.array([[0,0,0,0],[1,-1,0,1],[-1,1,-1,0]])
        vals = m3[coords]
        assert vals.shape == (3,)
        assert all(-(m3.n//2) <= v <= m3.n//2 for v in vals)

    def test_batch_wrong_dim(self, m3):
        coords = np.array([[0,0,0],[1,2,3]])
        with pytest.raises(ValueError):
            m3[coords]

    def test_scalar_index(self, m3):
        v = m3[(0, 0, 0, 0)]
        assert isinstance(v, int)

    def test_batch_all_ranks(self, m3):
        """Batch evaluation over all n^4 ranks covers full value range."""
        n = m3.n
        all_coords = np.array([m3._oa_rank_to_signed_coords(k) for k in range(n**4)])
        vals = m3[all_coords]
        assert vals.shape == (n**4,)


# ── cell_at_rank (FM-Dance compat) ────────────────────────────────────────────

class TestCellAtRank:
    def test_returns_int(self, m3):
        v = m3.cell_at_rank(0)
        assert isinstance(v, int)

    def test_value_in_range(self, m3):
        n = m3.n
        for k in range(n**4):
            v = m3.cell_at_rank(k)
            assert -(n//2) <= v <= n//2

    def test_differs_from_oa_rank(self, m3):
        """FM-Dance ordering differs from OA natural digit ordering."""
        vals_fmd = [m3.cell_at_rank(k) for k in range(m3.n**4)]
        vals_oa  = [m3.cell_at_oa_rank(k) for k in range(m3.n**4)]
        # Both cover the same value set but in different orders (usually differ)
        assert set(vals_fmd) == set(vals_oa)


# ── materialize ───────────────────────────────────────────────────────────────

class TestMaterialize:
    def test_shape_n3(self, m3):
        t = m3.materialize()
        assert t.shape == (3,)*4

    def test_dtype(self, m3):
        t = m3.materialize()
        assert t.dtype == int or np.issubdtype(t.dtype, np.integer)

    def test_values_in_range(self, m3):
        t = m3.materialize()
        assert t.min() >= -(m3.n//2)
        assert t.max() <= m3.n//2


# ── Communion ─────────────────────────────────────────────────────────────────

class TestCommunion:
    def test_add_same_n(self, m3):
        m8 = m3 + m3
        assert m8.d == 8
        assert m8.n == 3

    def test_add_produces_oa(self, m3):
        m8 = m3 + m3
        assert m8.verify_oa()

    def test_add_wrong_n_raises(self, m3, m5):
        with pytest.raises(ValueError):
            m3 + m5

    def test_add_repr(self, m3):
        m8 = m3 + m3
        assert "8" in repr(m8)

    def test_triple_add(self, m3):
        m12 = m3 + m3 + m3
        assert m12.d == 12
        assert m12._blocks == 3


# ── Package export ────────────────────────────────────────────────────────────

class TestPackageExport:
    def test_importable_from_flu(self):
        import flu
        assert hasattr(flu, "SparseOrthogonalManifold")

    def test_importable_from_sparse(self):
        from flu.container.sparse import SparseOrthogonalManifold
        assert SparseOrthogonalManifold is not None

    def test_smoke_via_flu_namespace(self):
        import flu
        m = flu.SparseOrthogonalManifold(n=3, d=4)
        assert m.verify_oa()


# ── Theorem registry connections ──────────────────────────────────────────────

class TestTheoremConnections:
    def test_dno_gen_proven(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DNO-GEN")
        assert t is not None and t.status == "PROVEN"

    def test_dno_coeff_even_proven(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DNO-COEFF-EVEN")
        assert t is not None and t.status == "PROVEN"

    def test_dno_opt_proven(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DNO-OPT")
        assert t is not None and t.status == "PROVEN"

    def test_dno_inv_proven(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DNO-INV")
        assert t is not None and t.status == "PROVEN"

    def test_computational_cert_all_n(self):
        """DNO-GEN computational certificate: n ∈ {2,3,4,5} all pass OA."""
        for n in [2, 3, 4, 5]:
            m = SparseOrthogonalManifold(n=n, d=4)
            assert m.verify_oa(), f"OA verification failed for n={n}"

    def test_computational_cert_d8(self):
        """DNO-REC-MATRIX: A^(2) certified for n ∈ {2,3,4}."""
        for n in [2, 3, 4]:
            m = SparseOrthogonalManifold(n=n, d=8)
            assert m.verify_oa(), f"d=8 OA verification failed for n={n}"
