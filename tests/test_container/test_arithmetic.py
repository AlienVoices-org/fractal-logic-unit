"""
tests/test_container/test_arithmetic.py
=======================================
Verification of OPER-1: Sparse Fractal Arithmetic.

Verifies:
  1. Lazy Evaluation (O(1) construction vs O(N) materialization)
  2. Associativity of (A+B)+C == A+(B+C)
  3. Scalar broadcasting (M + c)
  4. Identity preservation (M + 0 == M)
  5. Multiplicative scaling (M * c)
  6. Compositional Correctness (multi-layer operator trees)
"""
import pytest
import numpy as np
import time
from flu.container.sparse import SparseCommunionManifold

# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def manifold():
    n, d = 5, 3
    seeds = [np.array([0, 1, 2]), np.array([2, 0, 1])]
    return SparseCommunionManifold(n=n, seeds=seeds)

# ── 1. Lazy Evaluation (O(1) construction) ───────────────────────────────────

def test_arithmetic_is_lazy(manifold):
    """
    Constructing (M+M) should be O(1) time (no grid allocation).
    Materializing should be O(n^d) time.
    """
    t0 = time.perf_counter()
    M_plus = manifold + manifold
    t1 = time.perf_counter()
    assert (t1 - t0) < 0.001, "Arithmetic construction must be lazy/O(1)"

    t0 = time.perf_counter()
    _ = M_plus.materialize()
    t1 = time.perf_counter()
    assert (t1 - t0) > 0.0001, "Materialization must be O(N)"

# ── 2. Associativity & Identity ──────────────────────────────────────────────

def test_addition_associativity(manifold):
    """(A + B) + C == A + (B + C)"""
    M1, M2, M3 = manifold, manifold, manifold
    lhs = (M1 + M2) + M3
    rhs = M1 + (M2 + M3)
    
    # Check on a sample coordinate
    coord = (0, 0)
    assert abs(lhs[coord] - rhs[coord]) < 1e-12

def test_add_identity(manifold):
    """M + 0 == M"""
    M = manifold
    M_plus = M + 0
    coord = (1, -1)
    assert abs(M_plus[coord] - M[coord]) < 1e-12

def test_mul_identity(manifold):
    """M * 1 == M"""
    M = manifold
    M_mul = M * 1.0
    coord = (-1, 0)
    assert abs(M_mul[coord] - M[coord]) < 1e-12

# ── 3. Scalar Broadcasting ───────────────────────────────────────────────────

def test_scalar_add_broadcast(manifold):
    """M + 5.0 applied to every cell."""
    M = manifold
    M_plus = M + 5.0
    coord = (0, 1)
    assert abs(M_plus[coord] - (M[coord] + 5.0)) < 1e-12

def test_scalar_mul_broadcast(manifold):
    """M * 2.0 applied to every cell."""
    M = manifold
    M_mul = M * 2.0
    coord = (-1, -1)
    assert abs(M_mul[coord] - (M[coord] * 2.0)) < 1e-12

# ── 4. Compositional Correctness ─────────────────────────────────────────────

def test_multi_layer_operator_tree(manifold):
    """Verify evaluation of (M + M) * 2.0 - M."""
    M = manifold
    # Tree: ( (M+M) * 2 ) - M
    expr = ((M + M) * 2.0) - M
    
    # Eval M + M = 2M; 2M * 2 = 4M; 4M - M = 3M
    coord = (1, 1)
    expected = 3 * M[coord]
    assert abs(expr[coord] - expected) < 1e-12

def test_materialize_equals_manual_iteration(manifold):
    """Materialize() must produce same results as per-cell iteration."""
    expr = (manifold + manifold) * 0.5
    mat  = expr.materialize()
    
    for idx in np.ndindex(*mat.shape):
        coord = tuple(i - manifold.n // 2 for i in idx)
        assert abs(mat[idx] - expr[coord]) < 1e-12

# ── 5. Division Safety ──────────────────────────────────────────────────────

def test_division_by_scalar(manifold):
    """M / 2.0."""
    M = manifold
    M_div = M / 2.0
    coord = (0, 0)
    assert abs(M_div[coord] - (M[coord] / 2.0)) < 1e-12
