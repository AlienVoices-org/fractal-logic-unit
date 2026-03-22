"""
tests/test_applications/test_hadamard.py
=========================================
Computational verification tests for HAD-1 (Walsh-Hadamard Generation).

These tests constitute the computational proof tier of HAD-1 (PROVEN,
algebraic_and_computational). They verify the core claim:

    H @ H.T == N * I   for all d in {2, 3, 4, 5, 6}

where N = 2^d and H is generated via bit-masked XOR-Communion seeds.

V15 — audit integration sprint.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from flu.applications.hadamard import HadamardGenerator


class TestHadamardOrthogonality(unittest.TestCase):
    """
    HAD-1 PROVEN: H @ H.T must equal exactly N * I.

    This is the core orthogonality condition for Hadamard matrices.
    """

    def setUp(self):
        self.gen = HadamardGenerator()

    def _check_d(self, d: int):
        N = 2 ** d
        H = self.gen.generate(d)
        ortho = H @ H.T
        expected = N * np.eye(N, dtype=np.int32)
        self.assertTrue(
            np.array_equal(ortho, expected),
            f"H @ H.T ≠ {N}·I for d={d}.\n"
            f"Max off-diagonal: {np.max(np.abs(ortho - expected))}",
        )

    def test_orthogonality_d2(self):
        """d=2: 4×4 Hadamard satisfies H @ H.T == 4·I."""
        self._check_d(2)

    def test_orthogonality_d3(self):
        """d=3: 8×8 Hadamard satisfies H @ H.T == 8·I."""
        self._check_d(3)

    def test_orthogonality_d4(self):
        """d=4: 16×16 Hadamard satisfies H @ H.T == 16·I."""
        self._check_d(4)

    def test_orthogonality_d5(self):
        """d=5: 32×32 Hadamard satisfies H @ H.T == 32·I."""
        self._check_d(5)

    def test_orthogonality_d6(self):
        """d=6: 64×64 Hadamard satisfies H @ H.T == 64·I. (Audit benchmark)"""
        self._check_d(6)


class TestHadamardStructure(unittest.TestCase):
    """Structural tests: entries ±1, row norms, row independence."""

    def setUp(self):
        self.gen = HadamardGenerator()

    def test_entries_are_plus_minus_one(self):
        """All entries of H must be exactly +1 or −1."""
        for d in [2, 3, 4]:
            H = self.gen.generate(d)
            unique = set(np.unique(H).tolist())
            self.assertEqual(
                unique, {-1, 1},
                f"d={d}: unexpected entries {unique - {-1, 1}}",
            )

    def test_row_norms_equal_sqrt_N(self):
        """Each row must have L2 norm = √N (follows from orthogonality)."""
        for d in [2, 3, 4]:
            N = 2 ** d
            H = self.gen.generate(d)
            norms = np.linalg.norm(H.astype(float), axis=1)
            expected = np.full(N, np.sqrt(N))
            np.testing.assert_allclose(norms, expected, atol=1e-9,
                err_msg=f"Row norms wrong for d={d}")

    def test_shape_is_N_by_N(self):
        """Generated matrix shape must be (2^d, 2^d)."""
        for d in [2, 3, 5]:
            N = 2 ** d
            H = self.gen.generate(d)
            self.assertEqual(H.shape, (N, N))

    def test_first_row_all_ones(self):
        """Row k=0 has all +1 (k_bits all 0 → dot product 0 → (−1)^0 = 1)."""
        for d in [2, 3, 4]:
            N = 2 ** d
            row0 = self.gen.generate_row(0, d)
            np.testing.assert_array_equal(
                row0, np.ones(N, dtype=np.int32),
                err_msg=f"d={d}: row 0 not all-ones",
            )

    def test_row_independence(self):
        """Rows must be linearly independent (matrix has full rank N)."""
        for d in [2, 3, 4]:
            H = self.gen.generate(d)
            rank = np.linalg.matrix_rank(H.astype(float))
            N = 2 ** d
            self.assertEqual(rank, N, f"d={d}: rank={rank} != N={N}")


class TestHadamardVerifyMethod(unittest.TestCase):
    """Test the .verify() convenience method."""

    def test_verify_returns_true_for_valid_depths(self):
        gen = HadamardGenerator()
        for d in [2, 3, 4, 5, 6]:
            self.assertTrue(gen.verify(d), f"verify() failed for d={d}")

    def test_generate_row_consistency(self):
        """generate_row(k, d) must match the k-th row of generate(d)."""
        gen = HadamardGenerator()
        for d in [3, 4]:
            H = gen.generate(d)
            for k in range(0, 2 ** d, 3):   # sample every 3rd
                row = gen.generate_row(k, d)
                np.testing.assert_array_equal(
                    row, H[k],
                    err_msg=f"generate_row({k},{d}) mismatch",
                )

    def test_invalid_d_raises(self):
        gen = HadamardGenerator()
        with self.assertRaises(ValueError):
            gen.generate(0)


class TestHadamardRegistryLink(unittest.TestCase):
    """Verify HAD-1 is in the theorem registry with correct status/proof."""

    def test_had1_is_proven(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        self.assertIsNotNone(t, "HAD-1 not found in registry")
        self.assertEqual(t.status, "PROVEN",
            f"HAD-1 status should be PROVEN, got {t.status}")

    def test_had1_proof_tier_is_algebraic_and_computational(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        self.assertEqual(
            t.proof_status, "algebraic_and_computational",
            f"Expected algebraic_and_computational, got {t.proof_status}",
        )

    def test_had1_references_bit_masking(self):
        """The corrected proof must mention the bit-masked seed construction."""
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        proof_lower = t.proof.lower()
        # Must reference the key correction: bit-masked / k_a AND x
        self.assertTrue(
            any(kw in proof_lower for kw in ["bit-masked", "k_a", "masking", "k · x"]),
            "HAD-1 proof does not reference the corrected bit-masked construction",
        )


if __name__ == "__main__":
    unittest.main()
