"""
Tests for flu.container.communion — CommunionEngine.

Key checks:
  - Non-associative phi is rejected at construction time.
  - Outer product shape is A.shape + B.shape.
  - Latin property propagates through outer communion (add phi).
  - Frozen phi association check works.
"""

import pytest
import numpy as np
from flu.container.communion import CommunionEngine
from flu.core.fm_dance       import generate_fast


def _make_latin(n, d):
    return generate_fast(n, d, signed=True).astype(np.int64)


class TestConstruction:
    def test_builtin_phi_accepted(self):
        for phi in ("add", "xor", "multiply"):
            eng = CommunionEngine(phi=phi)
            assert eng is not None

    def test_custom_associative_phi(self):
        eng = CommunionEngine(phi=lambda a, b: a + b)
        assert eng is not None

    def test_non_associative_phi_rejected(self):
        """Subtraction is not associative: (a-b)-c ≠ a-(b-c)."""
        with pytest.raises(ValueError, match="associativity"):
            CommunionEngine(phi=lambda a, b: a - b)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            CommunionEngine(mode="quantum_teleport")


class TestOuterProduct:
    def test_shape(self):
        A   = _make_latin(3, 2)
        B   = _make_latin(3, 2)
        eng = CommunionEngine(phi="add", mode="outer")
        C   = eng.commune(A, B)
        assert C.shape == A.shape + B.shape   # (3,3,3,3)

    def test_shape_different_dims(self):
        A   = _make_latin(3, 2)   # shape (3,3)
        B   = _make_latin(3, 3)   # shape (3,3,3)
        eng = CommunionEngine(phi="add", mode="outer")
        C   = eng.commune(A, B)
        assert C.shape == (3, 3, 3, 3, 3)

    def test_latin_propagates(self):
        """Latin A and B → C should have 3^3 unique values per slice (add phi)."""
        from flu.utils.verification import check_latin
        A   = _make_latin(3, 2)
        B   = _make_latin(3, 2)
        eng = CommunionEngine(phi="add", mode="outer")
        C   = eng.commune(A, B)
        # The combined digit range is [-4,4] (add of two [-1,1] arrays)
        # Check that axis-0 slices have exactly 3 distinct values each
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = set(int(v) for v in C[i, j, k, :].flatten())
                    assert len(s) == 3


class TestKronecker:
    def test_shape(self):
        A   = _make_latin(3, 2)   # shape (3,3)
        B   = _make_latin(3, 2)   # shape (3,3)
        eng = CommunionEngine(phi="add", mode="kronecker")
        C   = eng.commune(A, B)
        assert C.shape == (3, 3, 3, 3)


class TestDirectSum:
    def test_shape(self):
        A   = _make_latin(3, 2)
        B   = _make_latin(3, 2)
        eng = CommunionEngine(phi="add", mode="direct")
        C   = eng.commune(A, B)
        assert C.shape == (3, 3, 3, 3)

    def test_mismatched_n_raises(self):
        A   = _make_latin(3, 2)
        B   = _make_latin(5, 2)
        eng = CommunionEngine(phi="add", mode="direct")
        with pytest.raises(ValueError, match="equal leading"):
            eng.commune(A, B)
