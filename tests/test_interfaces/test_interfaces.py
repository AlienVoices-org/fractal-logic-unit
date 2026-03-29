"""
tests/test_interfaces/test_interfaces.py
=============================================
Test suite for FLU interface facets and new bridge theorems.

Coverage:
  - LexiconFacet  (LEX-1  PROVEN)
  - IntegrityFacet (INT-1  PROVEN)
  - GeneticFacet  (GEN-1  PROVEN)
  - InvarianceFacet (INV-1 PROVEN)
  - HilbertFacet  (HIL-1  RETIRED in V15.1.3 — preserved for backward compat)
  - CohomologyFacet (DEC-1 PROVEN in V15.1.2)
  - HAD-1 theorem registration
  - TSP-1 theorem registration
  - CRYPTO-1 theorem registration

V15.3 current.
"""

import sys
import os
import unittest

import numpy as np

# Ensure src/ on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ── LexiconFacet ─────────────────────────────────────────────────────────────

class TestLexiconFacet(unittest.TestCase):

    def setUp(self):
        from flu.interfaces.lexicon import LexiconFacet
        self.LF = LexiconFacet

    def test_encode_decode_roundtrip_n3_d4(self):
        lex = self.LF(n=3, d=4)
        # Use unsigned coords (0..n-1) for clean roundtrip
        for coord in [(0,0,0,0), (1,2,0,1), (2,2,2,2)]:
            self.assertEqual(lex.decode(lex.encode(coord)), coord)

    def test_encode_decode_roundtrip_n2_d8(self):
        lex = self.LF(n=2, d=8)
        coord = (0, 1, 0, 1, 1, 0, 1, 0)
        self.assertEqual(lex.decode(lex.encode(coord)), coord)

    def test_all_symbols_unique(self):
        lex = self.LF(n=3, d=3)
        symbols = lex.all_symbols()
        self.assertEqual(len(symbols), 27)
        self.assertEqual(len(set(symbols)), 27)

    def test_encode_rank(self):
        lex = self.LF(n=3, d=2)
        s0 = lex.encode_rank(0)
        self.assertEqual(len(s0), 2)

    def test_wrong_length_raises(self):
        lex = self.LF(n=3, d=3)
        with self.assertRaises(ValueError):
            lex.encode((0, 1))  # too short

    def test_facet_info(self):
        lex = self.LF(n=3, d=4)
        info = lex.info()
        self.assertEqual(info.theorem_id, "LEX-1")
        self.assertEqual(info.status, "PROVEN")

    def test_custom_alphabet(self):
        lex = self.LF(n=3, d=2, alphabet="XYZ")
        encoded = lex.encode((1, 2))
        self.assertEqual(encoded, "YZ")
        self.assertEqual(lex.decode(encoded), (1, 2))


# ── IntegrityFacet ────────────────────────────────────────────────────────────

class TestIntegrityFacet(unittest.TestCase):

    def setUp(self):
        import flu
        from flu.interfaces.integrity import IntegrityFacet
        self.M3 = flu.generate(3, 4)
        self.sonde3 = IntegrityFacet(self.M3, n=3, signed=True)

    def test_l1_check_passes_for_valid_manifold(self):
        ok, detail = self.sonde3.check_line(coord=(0, 0, 0, 0), axis=0)
        self.assertTrue(ok, f"L1 check should pass: {detail}")

    def test_l1_check_all_axes_at_origin(self):
        all_ok, details = self.sonde3.check_all_lines_at((0, 0, 0, 0))
        self.assertTrue(all_ok, f"All L1 checks at origin should pass: {details}")

    def test_l1_check_interior_coordinate(self):
        ok, _ = self.sonde3.check_line(coord=(1, 2, 0, 1), axis=2)
        self.assertTrue(ok)

    def test_facet_info(self):
        info = self.sonde3.info()
        self.assertEqual(info.theorem_id, "INT-1")
        self.assertEqual(info.status, "PROVEN")

    def test_corrupted_manifold_detected(self):
        """Deliberately corrupt the manifold and verify detection."""
        M_bad = self.M3.copy()
        M_bad[0, 0, 0, 0] += 1  # break L1
        from flu.interfaces.integrity import IntegrityFacet
        sonde_bad = IntegrityFacet(M_bad, n=3, signed=True)
        all_ok, _ = sonde_bad.check_all_lines_at((0, 0, 0, 0))
        # At least one axis should fail now
        self.assertFalse(all_ok)


# ── GeneticFacet ──────────────────────────────────────────────────────────────

class TestGeneticFacet(unittest.TestCase):

    def setUp(self):
        from flu.interfaces.genetic import GeneticFacet
        self.gf = GeneticFacet(populate_from_golden=True)

    def test_facet_info(self):
        info = self.gf.info()
        self.assertEqual(info.theorem_id, "GEN-1")
        self.assertEqual(info.status, "PROVEN")

    def test_available_n_non_empty(self):
        ns = self.gf.available_n()
        self.assertGreater(len(ns), 0)

    def test_sha256_verify_all(self):
        results = self.gf.verify_all()
        for n, ok in results.items():
            self.assertTrue(ok, f"SHA-256 verification failed for n={n}")

    def test_get_returns_record(self):
        ns = self.gf.available_n()
        n = ns[0]
        rec = self.gf.get(n)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.n, n)

    def test_export_import_roundtrip(self):
        blob = self.gf.export_json()
        from flu.interfaces.genetic import GeneticFacet
        gf2 = GeneticFacet(populate_from_golden=False)
        gf2.import_json(blob)
        results = gf2.verify_all()
        for n, ok in results.items():
            self.assertTrue(ok, f"Post-import SHA-256 fail for n={n}")

    def test_add_custom_seed(self):
        perm = list(range(5))  # identity for n=5
        rec = self.gf.add(n=5, perm=perm, provenance="test")
        self.assertTrue(rec.verify())
        self.assertEqual(rec.n, 5)

    def test_sha256_mismatch_raises(self):
        from flu.interfaces.genetic import SeedRecord
        with self.assertRaises(ValueError):
            SeedRecord(n=3, perm=[0, 1, 2], delta=3,
                       sha256="0" * 64, provenance="bad")


# ── InvarianceFacet ───────────────────────────────────────────────────────────

class TestInvarianceFacet(unittest.TestCase):

    def setUp(self):
        from flu.interfaces.invariance import InvarianceFacet
        self.inv = InvarianceFacet(n=3, d=3)

    def test_facet_info(self):
        info = self.inv.info()
        self.assertEqual(info.theorem_id, "INV-1")
        self.assertEqual(info.status, "PROVEN")

    def test_t3_passes_for_flu_manifold(self):
        import flu
        M = flu.generate(3, 3)
        self.assertTrue(self.inv.check_t3(M))

    def test_l1_passes_for_flu_manifold(self):
        import flu
        M = flu.generate(3, 3)
        self.assertTrue(self.inv.check_l1(M))

    def test_compare_branches_both_have_t3(self):
        report = self.inv.compare_branches()
        self.assertTrue(report["odd_branch"]["T3"],
                        "FM-Dance (odd) should satisfy T3")
        self.assertTrue(report["even_branch"]["T3"],
                        "Sum-Mod (even) should satisfy T3")

    def test_compare_branches_match(self):
        report = self.inv.compare_branches()
        self.assertTrue(report["all_invariants_match"],
                        f"Branches should match: {report}")


# ── HilbertFacet ──────────────────────────────────────────────────────────────
# HIL-1 is RETIRED (V15.1.3). Tests preserved for backward-compat verification.
# HilbertFacet emits DeprecationWarning on construction — suppressed in tests.

class TestHilbertFacet(unittest.TestCase):

    def test_facet_info(self):
        import warnings
        from flu.interfaces.hilbert import HilbertFacet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hf = HilbertFacet(d=2, n=3)
        info = hf.info()
        self.assertEqual(info.theorem_id, "HIL-1")
        self.assertEqual(info.status, "RETIRED")  # HIL-1 RETIRED V15.1.3

    def test_get_point_returns_valid_coord(self):
        import warnings
        from flu.interfaces.hilbert import HilbertFacet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hf = HilbertFacet(d=2, n=3)
        for k in range(4):
            pt = hf.get_point(k)
            self.assertEqual(len(pt), 2)
            self.assertTrue(all(0 <= c < 3 for c in pt))

    def test_get_all_points_correct_count(self):
        import warnings
        from flu.interfaces.hilbert import HilbertFacet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hf = HilbertFacet(d=3, n=3)
        pts = hf.get_all_points()
        self.assertEqual(len(pts), 27)

    def test_untuned_is_plain_fmdance(self):
        """With tune=False, HilbertFacet should match unsigned path_coord."""
        import warnings
        from flu.interfaces.hilbert import HilbertFacet
        from flu.core.fm_dance_path import path_coord
        n = 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hf = HilbertFacet(d=3, n=n, tune=False)
        for k in range(27):
            # HilbertFacet returns unsigned [0,n); path_coord returns signed
            expected_unsigned = np.array([int(c) % n for c in path_coord(k, n, 3)])
            np.testing.assert_array_equal(
                hf.get_point(k),
                expected_unsigned,
            )

    def test_locality_score_returns_float(self):
        import warnings
        from flu.interfaces.hilbert import HilbertFacet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hf = HilbertFacet(d=2, n=3)
        score = hf.locality_score()
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_rotation_hub(self):
        from flu.interfaces.hilbert import RotationHub
        hub = RotationHub(d=3)
        coords = np.array([0, 1, 2])
        result = hub.apply_at_carry(coords, j=1, n=3)
        self.assertEqual(len(result), 3)

    def test_emits_deprecation_warning(self):
        """HilbertFacet must emit DeprecationWarning on construction (RETIRED)."""
        import warnings
        from flu.interfaces.hilbert import HilbertFacet
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HilbertFacet(d=2, n=3)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertGreater(len(dep_warnings), 0, "Expected DeprecationWarning for RETIRED HilbertFacet")


# ── CohomologyFacet ───────────────────────────────────────────────────────────

class TestCohomologyFacet(unittest.TestCase):

    def setUp(self):
        from flu.interfaces.cohomology import CohomologyFacet
        import flu
        self.M = flu.generate(3, 3)
        self.coh = CohomologyFacet(n=3, d=3)

    def test_facet_info(self):
        info = self.coh.info()
        self.assertEqual(info.theorem_id, "DEC-1")
        self.assertEqual(info.status, "PROVEN")  # DEC-1 promoted in V15.1.2

    def test_coboundary_shape(self):
        dM = self.coh.coboundary(self.M)
        self.assertEqual(dM.shape, (3, 3, 3, 3))  # (n,n,n,d)

    def test_coboundary_zero_sum_on_torus(self):
        """The sum of axis-0 coboundary should be zero (toroidal cancellation)."""
        dM = self.coh.coboundary(self.M)
        for axis in range(3):
            total = np.sum(dM[..., axis])
            self.assertAlmostEqual(float(total), 0.0, places=10)

    def test_laplacian_shape(self):
        lap = self.coh.laplacian(self.M.astype(float))
        self.assertEqual(lap.shape, self.M.shape)

    def test_circulation_zero_for_empty_scars(self):
        loop = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        circ = self.coh.circulation({}, loop)
        self.assertEqual(circ, 0.0)

    def test_circulation_nonzero_for_scar(self):
        scars = {(0, 0, 0): 1.5, (1, 0, 0): 0.5}
        loop = [(0, 0, 0), (1, 0, 0)]
        circ = self.coh.circulation(scars, loop)
        self.assertAlmostEqual(circ, 2.0)

    def test_homology_class_returns_dict(self):
        hom = self.coh.homology_class({})
        self.assertEqual(len(hom), 3)


# ── New Bridge Theorems in Registry ──────────────────────────────────────────

class TestV15BridgeTheorems(unittest.TestCase):

    def test_had1_in_registry(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        self.assertIsNotNone(t)
        self.assertEqual(t.status, "PROVEN")
        self.assertIn("Hadamard", t.name)

    def test_tsp1_in_registry(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("TSP-1")
        self.assertIsNotNone(t)
        self.assertEqual(t.status, "PROVEN")
        self.assertIn("TSP", t.name)

    def test_crypto1_in_registry(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("CRYPTO-1")
        self.assertIsNotNone(t)
        self.assertEqual(t.status, "PROVEN")
        self.assertIn("APN", t.name)

    def test_proven_count_is_59(self):
        from flu.theory.theorem_registry import proven_theorems
        pt = proven_theorems()
        self.assertEqual(len(pt), 71,
            f"Expected 71 PROVEN in V15.3.2 (DN1-GEN+DN1-REC added), got {len(pt)}")

    def test_total_count_is_65(self):
        from flu.theory.theorem_registry import REGISTRY
        self.assertEqual(len(REGISTRY), 75,
            f"Expected 75 total in V15.3.2 (DN1-GEN+DN1-REC added), got {len(REGISTRY)}")

    def test_conjecture_count_unchanged(self):
        from flu.theory.theorem_registry import open_conjectures
        conjs = open_conjectures()
        self.assertEqual(len(conjs), 2,
            f"Open items should be 2 (OD-16, OD-17; DN1 now PROVEN), got {len(conjs)}: {[c.name for c in conjs]}")

    def test_had1_references_corrected_construction(self):
        """HAD-1 (corrected V15) must reference the bit-masked construction, not just PC-2."""
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        refs_str = " ".join(t.references).lower()
        # Corrected proof references HadamardGenerator and audit notes, not the flawed PC-2 path
        self.assertTrue(
            "hadamard" in refs_str or "audit" in refs_str or "sylvester" in refs_str,
            f"HAD-1 references don't mention the corrected construction: {t.references}",
        )

    def test_tsp1_conditions(self):
        """TSP-1 must state the uniform Cayley graph condition."""
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("TSP-1")
        conditions_str = " ".join(t.conditions)
        self.assertIn("uniform", conditions_str.lower())

    def test_crypto1_conditions_mention_apn(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("CRYPTO-1")
        conditions_str = " ".join(t.conditions)
        self.assertIn("APN", conditions_str)


# ── Interfaces package import ─────────────────────────────────────────────────

class TestInterfacesPackage(unittest.TestCase):

    def test_all_facets_importable(self):
        from flu.interfaces import (
            FluFacet, LexiconFacet, IntegrityFacet, GeneticFacet,
            InvarianceFacet, HilbertFacet, CohomologyFacet, GrayCodeFacet,
        )  # HilbertFacet is RETIRED but still importable for backward compat

    def test_facets_have_correct_statuses(self):
        import warnings
        from flu.interfaces import (
            LexiconFacet, IntegrityFacet, GeneticFacet,
            InvarianceFacet, HilbertFacet, CohomologyFacet, GrayCodeFacet,
        )
        lex = LexiconFacet(3, 2)
        self.assertEqual(lex.status, "PROVEN")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hf = HilbertFacet(d=2, n=3)
        self.assertEqual(hf.status, "RETIRED")  # HIL-1 RETIRED V15.1.3
        coh = CohomologyFacet(n=3, d=3)
        self.assertEqual(coh.status, "PROVEN")  # DEC-1 proven in V15.1.2
        gc = GrayCodeFacet(d=4, n=2)
        self.assertEqual(gc.status, "PROVEN")  # T8 is PROVEN since V13

    def test_gray_code_facet_importable(self):
        from flu.interfaces import GrayCodeFacet, binary_gray_encode, binary_gray_decode
        self.assertIsNotNone(GrayCodeFacet)
        self.assertIsNotNone(binary_gray_encode)
        self.assertIsNotNone(binary_gray_decode)


# ── GrayCodeFacet tests ───────────────────────────────────────────────────────

class TestGrayCodeFacetBinary(unittest.TestCase):
    """T8 computational verification for n=2 binary Gray codes."""

    def setUp(self):
        from flu.interfaces.gray_code import GrayCodeFacet
        self.GC = GrayCodeFacet

    def test_gray_property_d2(self):
        """All consecutive codewords differ by Hamming 1 for d=2."""
        gc = self.GC(d=2, n=2)
        self.assertTrue(gc.verify_gray_property())

    def test_gray_property_d3(self):
        """All consecutive codewords differ by Hamming 1 for d=3."""
        gc = self.GC(d=3, n=2)
        self.assertTrue(gc.verify_gray_property())

    def test_gray_property_d4(self):
        """All consecutive codewords differ by Hamming 1 for d=4."""
        gc = self.GC(d=4, n=2)
        self.assertTrue(gc.verify_gray_property())

    def test_hamiltonian_d4(self):
        """Full cycle visits all N=16 distinct codewords."""
        gc = self.GC(d=4, n=2)
        result = gc.verify_t8_computational()
        self.assertTrue(result["hamiltonian_ok"])
        self.assertEqual(result["N"], 16)

    def test_wraparound_d3(self):
        """Last codeword → first codeword is distance 1 (toroidal closure)."""
        gc = self.GC(d=3, n=2)
        result = gc.verify_t8_computational()
        self.assertTrue(result["wraparound_ok"])

    def test_all_distances_one_d4(self):
        """Every step has Hamming distance exactly 1."""
        gc = self.GC(d=4, n=2)
        dists = gc.hamming_distances()
        self.assertTrue(
            np.all(dists == 1),
            f"Some distances ≠ 1: {dists[dists != 1]}",
        )

    def test_first_codeword_is_zero(self):
        """Rank 0 → all-zeros codeword (G(0) = 0)."""
        gc = self.GC(d=4, n=2)
        cw = gc.get_codeword(0)
        np.testing.assert_array_equal(cw, np.zeros(4, dtype=np.int8))

    def test_sequence_length(self):
        """sequence() returns exactly N codewords."""
        for d in [2, 3, 4]:
            gc = self.GC(d=d, n=2)
            seq = gc.sequence()
            self.assertEqual(len(seq), 2 ** d)

    def test_bit_trick_round_trip(self):
        """binary_gray_encode and binary_gray_decode are inverse."""
        from flu.interfaces.gray_code import binary_gray_encode, binary_gray_decode
        for k in range(64):
            self.assertEqual(binary_gray_decode(binary_gray_encode(k)), k)


class TestGrayCodeFacetNAry(unittest.TestCase):
    """T8 extension: n-ary Gray (odd n via FM-Dance)."""

    def setUp(self):
        from flu.interfaces.gray_code import GrayCodeFacet
        self.GC = GrayCodeFacet

    def test_gray_property_n3_d2(self):
        """n=3, d=2: all L_∞ distances ≤ 1."""
        gc = self.GC(d=2, n=3)
        result = gc.verify_t8_computational()
        self.assertTrue(result["gray_ok"], f"n-ary Gray failed: {result}")

    def test_hamiltonian_n3_d2(self):
        """n=3, d=2: visits all 9 distinct points."""
        gc = self.GC(d=2, n=3)
        result = gc.verify_t8_computational()
        self.assertTrue(result["hamiltonian_ok"])
        self.assertEqual(result["N"], 9)

    def test_facet_info(self):
        gc = self.GC(d=3, n=2)
        info = gc.info()
        self.assertIn("GrayCodeFacet", info.name)
        self.assertEqual(info.status, "PROVEN")  # T8 is PROVEN since V13

    def test_t8_registry_link(self):
        """GrayCodeFacet must link to theorem T8 (PROVEN since V13)."""
        gc = self.GC(d=2, n=2)
        from flu.theory.theorem_registry import get_theorem
        t8 = get_theorem("T8")
        self.assertIsNotNone(t8, "T8 not found in registry")
        self.assertEqual(t8.status, "PROVEN",
            "T8 (FM-Dance as BRGC) was proven in V13 — should be PROVEN")

    def test_invalid_n_raises(self):
        """Even n ≥ 4 raises ValueError."""
        with self.assertRaises(ValueError):
            self.GC(d=3, n=4)

    def test_d1_supported(self):
        """d=1 is the degenerate (trivial) case."""
        gc = self.GC(d=1, n=2)
        self.assertEqual(gc.N, 2)
        result = gc.verify_t8_computational()
        self.assertTrue(result["hamiltonian_ok"])


# ── HAD-1 corrected proof-tier check ─────────────────────────────────────────

class TestHAD1CorrectedProof(unittest.TestCase):
    """Verify HAD-1 uses the corrected bit-masked proof, not the flawed fold-XOR."""

    def test_had1_proof_status_is_algebraic_and_computational(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        self.assertEqual(t.proof_status, "algebraic_and_computational",
            "HAD-1 must be algebraic_and_computational (corrected in V15 sprint)")

    def test_had1_proof_mentions_bit_masked(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        proof_lower = t.proof.lower()
        self.assertTrue(
            "bit-masked" in proof_lower or "k_a" in proof_lower or "masking" in proof_lower,
            "HAD-1 proof must reference the corrected bit-masked seed construction",
        )

    def test_had1_proof_mentions_audit_correction(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        # The corrected proof must acknowledge the flaw that was caught
        self.assertTrue(
            "audit" in t.proof.lower() or "flaw" in t.proof.lower() or "corrected" in t.proof.lower(),
            "HAD-1 proof should acknowledge the V15 audit correction",
        )

    def test_had1_conditions_mention_parametrised_seeds(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("HAD-1")
        conditions_str = " ".join(t.conditions).lower()
        self.assertTrue(
            "bit" in conditions_str or "parametri" in conditions_str or "k_a" in conditions_str,
            "HAD-1 conditions should mention bit-masked / parametrised seeds",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── DISC-1: Discrete Integral Identity ────────────────────────────────────────

class TestDiscreteIntegralIdentity(unittest.TestCase):
    """
    Tests for DISC-1 — FM-Dance Discrete Integral Identity.
    Integrated from V15 MathReview audit.
    """

    def test_disc1_in_registry(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DISC-1")
        self.assertIsNotNone(t)
        self.assertEqual(t.status, "PROVEN")
        self.assertIn("Discrete Integral", t.name)

    def test_disc1_proof_status(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DISC-1")
        self.assertEqual(t.proof_status, "algebraic_and_computational")

    def test_phi_equals_T_times_digits_n3_d3(self):
        """Φ(k) = T·a(k): verify all k for n=3, d=3."""
        from flu.theory.theory_fm_dance import verify_discrete_integral_identity
        result = verify_discrete_integral_identity(n=3, d=3)
        self.assertTrue(result["phi_identity_ok"],
                        "Φ(k) = T·a(k) failed for n=3, d=3")

    def test_phi_equals_T_times_digits_n5_d2(self):
        from flu.theory.theory_fm_dance import verify_discrete_integral_identity
        result = verify_discrete_integral_identity(n=5, d=2)
        self.assertTrue(result["phi_identity_ok"],
                        "Φ(k) = T·a(k) failed for n=5, d=2")

    def test_step_vector_equals_T_times_delta_a_n3(self):
        """σ_j = T·Δa_j: verify all j for n=3, d=4."""
        from flu.theory.theory_fm_dance import verify_discrete_integral_identity
        result = verify_discrete_integral_identity(n=3, d=4)
        self.assertTrue(result["step_identity_ok"],
                        "σ_j = T·Δa_j failed for n=3, d=4")

    def test_step_vector_equals_T_times_delta_a_n7(self):
        from flu.theory.theory_fm_dance import verify_discrete_integral_identity
        result = verify_discrete_integral_identity(n=7, d=3)
        self.assertTrue(result["step_identity_ok"],
                        "σ_j = T·Δa_j failed for n=7, d=3")

    def test_t_inverse_is_forward_difference(self):
        """T^{-1} is bidiagonal forward-difference — NOT the Pascal inverse."""
        import numpy as np
        d = 5
        # Build FLU T
        T = np.zeros((d, d), dtype=int)
        T[0, 0] = -1
        for i in range(1, d):
            for j in range(i + 1):
                T[i, j] = 1
        T_inv = np.round(np.linalg.inv(T.astype(float))).astype(int)

        # Verify T·T^{-1} = I
        self.assertTrue(np.array_equal(T @ T_inv, np.eye(d, dtype=int)),
                        "T·T^{-1} ≠ I")

        # Verify NOT the Pascal inverse (reviewer confusion)
        from math import comb
        pascal_inv = np.array(
            [[(-1)**(i-j)*comb(i,j) if j<=i else 0 for j in range(d)]
             for i in range(d)], dtype=int)
        self.assertFalse(np.array_equal(T_inv, pascal_inv),
                         "T^{-1} should NOT equal Pascal inverse — they are different matrices")

    def test_disc1_references_kib(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DISC-1")
        refs = " ".join(t.references)
        self.assertIn("KIB", refs)

    def test_disc1_mentions_vdC_duality(self):
        from flu.theory.theorem_registry import get_theorem
        t = get_theorem("DISC-1")
        stmt = t.statement.lower()
        self.assertTrue("van der corput" in stmt or "radical" in stmt,
                        "DISC-1 should mention van der Corput duality")

    def test_disc1_importable_from_flu(self):
        import flu
        self.assertTrue(hasattr(flu, "DISC1_DISCRETE_INTEGRAL"))
        self.assertTrue(hasattr(flu, "verify_discrete_integral_identity"))
        self.assertTrue(callable(flu.verify_discrete_integral_identity))
