"""
tests/test_container/test_sparse_export.py
============================================
SparseCommunionManifold, flu.manifold() factory, n-ary cell limits,
and container export (to_numpy_buffer, fill_weight_matrix).

Theorems touched: T1 (Latin via sparse), HM-1 (holographic sparsity).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import unittest


# ── SparseCommunionManifold ───────────────────────────────────────────────────

class TestSparseCommunionManifold(unittest.TestCase):

    def test_import(self):
        from flu.container.sparse import SparseCommunionManifold
        self.assertIsNotNone(SparseCommunionManifold)

    def test_single_cell_evaluation(self):
        from flu.container.sparse import SparseCommunionManifold
        n, d = 3, 2
        seeds = [np.array([0, 1, 2]), np.array([2, 0, 1])]
        M = SparseCommunionManifold(n=n, seeds=seeds)
        val = M[0, -1]
        self.assertIsInstance(val, (int, np.integer))

    def test_shape_attribute(self):
        from flu.container.sparse import SparseCommunionManifold
        n, d = 5, 3
        M = SparseCommunionManifold(n=n, seeds=[np.arange(n)]*d)
        self.assertEqual(M.shape, (5, 5, 5))
        self.assertEqual(M.d, 3)
        self.assertEqual(M.n, 5)

    def test_even_n_raises(self):
        from flu.container.sparse import SparseCommunionManifold
        with self.assertRaises(ValueError):
            SparseCommunionManifold(n=4, seeds=[np.arange(4)])

    def test_wrong_seed_shape_raises(self):
        from flu.container.sparse import SparseCommunionManifold
        with self.assertRaises(ValueError):
            SparseCommunionManifold(n=3, seeds=[np.array([0, 1])])  # len 2 ≠ 3

    def test_batch_evaluation(self):
        from flu.container.sparse import SparseCommunionManifold
        n, d = 5, 2
        M = SparseCommunionManifold(n=n, seeds=[np.arange(n)]*d)
        coords = np.array([[-2, -2], [0, 0], [2, 2]])
        batch  = M[coords]
        for i, c in enumerate(coords):
            self.assertEqual(batch[i], M[tuple(c)])

    def test_communion_seed_concatenation(self):
        from flu.container.sparse import SparseCommunionManifold
        n = 3
        M1 = SparseCommunionManifold(n=n, seeds=[np.array([0,1,2]), np.array([2,0,1])])
        M2 = SparseCommunionManifold(n=n, seeds=[np.array([1,2,0])])
        MC = SparseCommunionManifold.commune(M1, M2)
        self.assertEqual(MC.d, 3)
        self.assertEqual(len(MC.seeds), 3)

    def test_communion_base_mismatch_raises(self):
        from flu.container.sparse import SparseCommunionManifold
        M1 = SparseCommunionManifold(n=3, seeds=[np.arange(3)])
        M2 = SparseCommunionManifold(n=5, seeds=[np.arange(5)])
        with self.assertRaises(ValueError):
            SparseCommunionManifold.commune(M1, M2)

    def test_cell_at_rank(self):
        """cell_at_rank must agree with __getitem__."""
        from flu.container.sparse import SparseCommunionManifold
        from flu.core.fm_dance import index_to_coords
        n, d = 3, 2
        M = SparseCommunionManifold(n=n, seeds=[np.array([0,1,2])]*d)
        for k in range(n**d):
            sc = tuple(int(c) for c in index_to_coords(k, n, d))
            self.assertEqual(M.cell_at_rank(k), M[sc])

    def test_latin_property_via_sparse(self):
        """T1/T3: values along any axis are a permutation of Z_n."""
        from flu.container.sparse import SparseCommunionManifold
        from flu.core.factoradic import unrank_optimal_seed
        n, d = 5, 2
        seeds = [unrank_optimal_seed(0, n, signed=False)]*d
        M = SparseCommunionManifold(n=n, seeds=seeds)
        half = n // 2
        for row in range(-half, half+1):
            vals = [M[row, col] for col in range(-half, half+1)]
            self.assertEqual(len(set(vals)), n)

    def test_repr(self):
        from flu.container.sparse import SparseCommunionManifold
        M = SparseCommunionManifold(n=3, seeds=[np.arange(3)])
        r = repr(M)
        self.assertIn("SparseCommunionManifold", r)
        self.assertIn("n=3", r)


# ── flu.manifold() factory ────────────────────────────────────────────────────

class TestManifoldFactory(unittest.TestCase):

    def test_dense_returns_ndarray(self):
        import flu
        M = flu.manifold(5, 2)
        self.assertIsInstance(M, np.ndarray)
        self.assertEqual(M.shape, (5, 5))

    def test_sparse_returns_sparse_manifold(self):
        import flu
        from flu.container.sparse import SparseCommunionManifold
        M = flu.manifold(3, 4, sparse=True)
        self.assertIsInstance(M, SparseCommunionManifold)
        self.assertEqual(M.d, 4)

    def test_sparse_large_d_is_cheap(self):
        """3^64 sparse manifold must be creatable in under 1 second."""
        import flu, time
        t0 = time.time()
        M  = flu.manifold(3, 64, sparse=True)
        self.assertLess(time.time() - t0, 1.0)
        self.assertEqual(M.d, 64)


# ── N-ary cell limits ─────────────────────────────────────────────────────────

class TestNaryMaxCells(unittest.TestCase):

    def test_default_limit_raises_on_large(self):
        from flu.core.n_ary import nary_generate
        with self.assertRaises(ValueError):
            nary_generate(11, 5)   # 11^5 = 161051 > 50_000

    def test_custom_limit_allows_larger(self):
        from flu.core.n_ary import nary_generate
        M = nary_generate(7, 3, max_cells=500)
        self.assertEqual(M.shape, (7, 7, 7))

    def test_inf_bypasses_limit(self):
        from flu.core.n_ary import nary_generate
        M = nary_generate(9, 3, max_cells=float("inf"))
        self.assertEqual(M.shape, (9, 9, 9))

    def test_signed_propagates_max_cells(self):
        from flu.core.n_ary import nary_generate_signed
        M = nary_generate_signed(7, 3, max_cells=500)
        self.assertEqual(M.shape, (7, 7, 7))


# ── container.export ─────────────────────────────────────────────────────────

class TestExportNumpy(unittest.TestCase):

    def _manifold(self, n=5, d=3):
        from flu.container.sparse import SparseCommunionManifold
        from flu.core.factoradic import unrank_optimal_seed
        seeds = [unrank_optimal_seed(0, n, signed=False)]*d
        return SparseCommunionManifold(n=n, seeds=seeds)

    def test_to_numpy_buffer_shape(self):
        from flu.container.export import to_numpy_buffer
        M = self._manifold(n=5, d=3)
        out = to_numpy_buffer(M)
        self.assertEqual(out.shape, M.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_to_numpy_buffer_zero_mean(self):
        from flu.container.export import to_numpy_buffer
        out = to_numpy_buffer(self._manifold(n=5, d=2), normalise=True)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)

    def test_to_numpy_buffer_value_range(self):
        from flu.container.export import to_numpy_buffer
        out = to_numpy_buffer(self._manifold(n=7, d=2), normalise=True)
        self.assertLessEqual(float(out.max()),  1.0 + 1e-5)
        self.assertGreaterEqual(float(out.min()), -1.0 - 1e-5)

    def test_to_numpy_buffer_inplace(self):
        from flu.container.export import to_numpy_buffer
        M   = self._manifold(n=3, d=2)
        buf = np.zeros(M.shape, dtype=np.float32)
        ret = to_numpy_buffer(M, out=buf)
        self.assertIs(ret, buf)
        self.assertFalse(np.all(buf == 0))

    def test_to_numpy_buffer_shape_mismatch_raises(self):
        from flu.container.export import to_numpy_buffer
        M = self._manifold(n=3, d=2)
        with self.assertRaises(ValueError):
            to_numpy_buffer(M, out=np.zeros((4, 4), dtype=np.float32))

    def test_to_numpy_buffer_unnormalised_range(self):
        from flu.container.export import to_numpy_buffer
        n = 5
        out = to_numpy_buffer(self._manifold(n=n, d=2), normalise=False, dtype=np.float32)
        half = n // 2
        self.assertLessEqual(float(out.max()),  half + 1e-5)
        self.assertGreaterEqual(float(out.min()), -half - 1e-5)

    def test_fill_weight_matrix_shape(self):
        from flu.container.export import fill_weight_matrix
        W = fill_weight_matrix(self._manifold(n=5, d=3), rows=5, cols=25)
        self.assertEqual(W.shape, (5, 25))

    def test_fill_weight_matrix_too_large_raises(self):
        from flu.container.export import fill_weight_matrix
        M = self._manifold(n=3, d=2)   # 9 cells
        with self.assertRaises(ValueError):
            fill_weight_matrix(M, rows=4, cols=4)  # 16 > 9

    def test_fill_weight_matrix_zero_mean(self):
        from flu.container.export import fill_weight_matrix
        W = fill_weight_matrix(self._manifold(n=5, d=3), rows=5, cols=25)
        self.assertAlmostEqual(float(W.mean()), 0.0, places=4)

    def test_export_module_importable(self):
        from flu.container import export
        for sym in ("to_numpy_buffer", "to_torch_buffer", "to_jax_buffer", "fill_weight_matrix"):
            self.assertTrue(hasattr(export, sym))

    def test_all_coords_covers_manifold(self):
        from flu.container.export import _all_coords
        n, d = 3, 3
        coords = _all_coords(n, d)
        self.assertEqual(coords.shape, (n**d, d))
        self.assertEqual(len(np.unique(coords, axis=0)), n**d)
