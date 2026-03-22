"""
tests/test_theory/test_theory.py
===============================
Tests for flu.theory — PhasedFractalNumberTheory and ModularLattice.

Covers previously untested:
  - PhasedFractalNumberTheory.get_container()
  - PhasedFractalNumberTheory.kinetic_completeness()
  - fm_dance_step_vectors()
  - fm_dance_forward()
  - verify_hamiltonian()
"""

import math
import pytest

from flu.theory.theory         import PhasedFractalNumberTheory
from flu.theory.theory_fm_dance import (
    ModularLattice,
    fm_dance_step_vectors,
    fm_dance_forward,
    verify_bijection,
    verify_hamiltonian,
)


# ── PhasedFractalNumberTheory.get_container ───────────────────────────────────

class TestGetContainer:

    @pytest.fixture(params=[3, 5, 7])
    def pfnt(self, request):
        return PhasedFractalNumberTheory(request.param, signed=True)

    def test_container_size_is_n_minus_1_factorial(self, pfnt):
        """THEOREM 1: |C_c| = (n-1)! for every pivot c."""
        n        = pfnt.n
        expected = math.factorial(n - 1)
        for pivot in pfnt.digits:
            arrows = pfnt.get_container(pivot)
            assert len(arrows) == expected, \
                f"n={n}, pivot={pivot}: got {len(arrows)}, expected {expected}"

    def test_centre_element_is_pivot(self, pfnt):
        """Every arrow in C_pivot has arrow[n//2] == pivot."""
        n = pfnt.n
        for pivot in pfnt.digits:
            for arrow in pfnt.get_container(pivot):
                assert arrow[n // 2] == pivot, \
                    f"Centre mismatch in arrow {arrow}, pivot={pivot}"

    def test_containers_are_disjoint(self, pfnt):
        """THEOREM 1: containers C_c are pairwise disjoint."""
        seen   = set()
        for pivot in pfnt.digits:
            for arrow in pfnt.get_container(pivot):
                key = tuple(arrow)
                assert key not in seen, f"Arrow {key} appears in multiple containers"
                seen.add(key)

    def test_union_covers_all_permutations(self, pfnt):
        """THEOREM 1: ∪ C_c = S_n(D)."""
        n        = pfnt.n
        expected = math.factorial(n)
        total    = sum(len(pfnt.get_container(p)) for p in pfnt.digits)
        assert total == expected

    def test_invalid_pivot_raises(self, pfnt):
        with pytest.raises(ValueError):
            pfnt.get_container(pivot=999)


# ── PhasedFractalNumberTheory.kinetic_completeness ────────────────────────────

class TestKineticCompleteness:

    def test_kinetic_completeness_n3(self):
        """THEOREM 4: kinetic_completeness returns a proven status dict."""
        pfnt   = PhasedFractalNumberTheory(3, signed=True)
        result = pfnt.kinetic_completeness()
        assert result["status"] == "PROVEN"
        assert result["container_size"] == math.factorial(3 - 1)   # (n-1)! = 2

    def test_kinetic_completeness_n5(self):
        pfnt   = PhasedFractalNumberTheory(5, signed=True)
        result = pfnt.kinetic_completeness()
        assert result["status"] == "PROVEN"
        assert result["container_size"] == math.factorial(4)   # (n-1)! = 24

    def test_kinetic_completeness_returns_dict(self):
        pfnt   = PhasedFractalNumberTheory(3, signed=True)
        result = pfnt.kinetic_completeness()
        assert isinstance(result, dict)
        assert "theorem" in result


# ── fm_dance_step_vectors ─────────────────────────────────────────────────────

class TestFmDanceStepVectors:

    @pytest.mark.parametrize("n,d", [(3, 2), (3, 4), (5, 2), (7, 3)])
    def test_returns_d_vectors(self, n, d):
        vecs = fm_dance_step_vectors(n, d)
        assert len(vecs) == d

    @pytest.mark.parametrize("n,d", [(3, 2), (5, 3)])
    def test_vectors_have_d_components(self, n, d):
        vecs = fm_dance_step_vectors(n, d)
        for v in vecs:
            assert len(v) == d

    def test_step_vectors_n3_d2_nonzero(self):
        """Step vectors must be non-zero for the traversal to work."""
        vecs = fm_dance_step_vectors(3, 2)
        for v in vecs:
            assert any(c != 0 for c in v), f"Zero step vector: {v}"


# ── fm_dance_forward ─────────────────────────────────────────────────────────

class TestFmDanceForward:

    @pytest.mark.parametrize("n,d", [(3, 2), (5, 2), (3, 3)])
    def test_output_in_valid_range(self, n, d):
        """All coordinates returned by fm_dance_forward are in [0, n)."""
        start = tuple([n // 2] * d)
        for k in range(n ** d):
            coord = fm_dance_forward(k, n, d, start)
            assert len(coord) == d
            for c in coord:
                assert 0 <= c < n, f"k={k}: coord {c} out of [0,{n})"

    def test_k0_returns_start(self):
        """k=0 should return the start position."""
        n, d  = 3, 2
        start = (1, 1)
        coord = fm_dance_forward(0, n, d, start)
        assert coord == start

    def test_bijection_consistency(self):
        """fm_dance_forward visits n^d distinct cells."""
        n, d  = 3, 2
        start = tuple([n // 2] * d)
        seen  = {fm_dance_forward(k, n, d, start) for k in range(n ** d)}
        assert len(seen) == n ** d


# ── verify_hamiltonian ────────────────────────────────────────────────────────

class TestVerifyHamiltonian:
    """
    THEOREM (Hamiltonian path), STATUS: PROVEN for standard FM-Dance step vectors.

    The FM-Dance step sequence visits every cell of the n^d torus exactly once,
    constituting a Hamiltonian path.  This is equivalent to the bijection theorem
    (no two consecutive steps land on the same cell), and is verified empirically
    here for the relevant (n, d) pairs used throughout FLU.
    """

    @pytest.mark.parametrize("n,d", [
        (3, 1), (3, 2), (3, 3), (3, 4),
        (5, 2), (5, 3),
        (7, 2),
    ])
    def test_hamiltonian_holds(self, n, d):
        """No two consecutive steps in fm_dance_forward land on the same cell."""
        assert verify_hamiltonian(n, d), \
            f"Hamiltonian property failed for n={n}, d={d}"

    @pytest.mark.parametrize("n,d", [(3, 2), (5, 2), (3, 4)])
    def test_hamiltonian_implies_bijection(self, n, d):
        """Hamiltonian ↔ bijection (same verification, different frame)."""
        assert verify_bijection(n, d)
        assert verify_hamiltonian(n, d)

    def test_hamiltonian_n3_d4_covers_all_81_cells(self):
        """n=3, d=4: the 3^4=81 cell torus has a Hamiltonian path."""
        assert verify_hamiltonian(3, 4)
