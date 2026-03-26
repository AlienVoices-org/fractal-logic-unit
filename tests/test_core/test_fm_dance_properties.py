"""
tests/test_core/test_fm_dance_properties.py
============================================
FM-Dance Kinetic Traversal — Rigorous Property Test Suite.

Tests the five proven properties of the FM-Dance path (flu.core.fm_dance_path):

    1. test_hamiltonian_coverage  – T2: visits every lattice point exactly once
    2. test_latin_projection      – T3: every 1-D slice is a permutation of Z_n
    3. test_step_bound            – T4: max torus step = min(d, floor(n/2))
    4. test_siamese_d2            – T5: D=2 reduces to classical Siamese algorithm
    5. test_torus_cycle           – C4: first and last points are within step bound

Also tests:
    6. test_bijection_roundtrip   – T1: path_coord / path_coord_to_rank are inverses
    7. test_mean_centered         – PFNT-2: column means = 0 (odd n, signed)
    8. test_fractal_low_dim       – T6: low d_split coords depend only on low digits
    9. test_theorem_registry      – All 11 proven theorems registered, 4 open
   10. test_addressing_vs_path    – fm_dance.py and fm_dance_path.py produce
                                    DIFFERENT traversal orders (they are distinct
                                    mathematical objects)

Runs without pytest: execute directly as  python -m tests.test_core.test_fm_dance_properties
or via unittest:  python -m unittest tests.test_core.test_fm_dance_properties
"""

from __future__ import annotations

import unittest
from typing import List, Tuple

import numpy as np

from flu.core.fm_dance_path import (
    path_coord,
    path_coord_to_rank,
    generate_path_array,
    verify_all,
    step_bound_theorem,
    verify_siamese_d2,
    verify_fractal,
)
from flu.core.fm_dance import (
    index_to_coords as addressing_coords,
    coords_to_index as addressing_rank,
)
from flu.theory.theorem_registry import (
    REGISTRY,
    proven_theorems,
    open_conjectures,
    get_theorem,
)
from flu.theory.theory_latin import (
    verify_holographic_repair,
    verify_constant_line_sum,
    verify_all_latin_theorems,
    line_sum_constant,
)
from flu.theory.theory_spectral import (
    verify_spectral_flatness,
    verify_dc_zero,
)
from flu.core.factoradic import factoradic_unrank


def build_value_hyperprism(n: int, d: int, signed: bool = True) -> np.ndarray:
    """
    Build the canonical value hyperprism: M[i_0,...,i_{d-1}] = (sum idx) mod n, signed.
    This is a Latin hyperprism with values in D_set, satisfying L1–L3 and S1–S2.
    """
    half = n // 2
    arr  = np.zeros([n] * d, dtype=float)
    for idx in np.ndindex(*[n] * d):
        val      = sum(idx) % n
        arr[idx] = val - half if signed else float(val)
    return arr


# ── Parametrised cases for the main property tests ───────────────────────────

CASES: List[Tuple[int, int]] = [
    (3, 1), (3, 2), (3, 3), (3, 4),
    (5, 2), (5, 3),
    (7, 2), (7, 3),
    (11, 2),
]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Bijection Round-Trip (Theorem T1)
# ─────────────────────────────────────────────────────────────────────────────

class TestBijectionRoundTrip(unittest.TestCase):
    """
    THEOREM T1 (n-ary Coordinate Bijection), STATUS: PROVEN.

    Every rank k maps to a unique coordinate, and the inverse mapping
    recovers k exactly.  Tests both directions of the bijection.
    """

    def _check_case(self, n: int, d: int) -> None:
        total = n ** d
        for k in range(total):
            c     = path_coord(k, n, d)
            k_inv = path_coord_to_rank(c, n, d)
            self.assertEqual(
                k_inv, k,
                f"Round-trip FAILED: n={n}, d={d}, k={k}, "
                f"coords={c}, k_back={k_inv}"
            )

    def test_roundtrip_n3_d1(self):  self._check_case(3, 1)
    def test_roundtrip_n3_d2(self):  self._check_case(3, 2)
    def test_roundtrip_n3_d3(self):  self._check_case(3, 3)
    def test_roundtrip_n3_d4(self):  self._check_case(3, 4)
    def test_roundtrip_n5_d2(self):  self._check_case(5, 2)
    def test_roundtrip_n5_d3(self):  self._check_case(5, 3)
    def test_roundtrip_n7_d2(self):  self._check_case(7, 2)
    def test_roundtrip_n7_d3(self):  self._check_case(7, 3)
    def test_roundtrip_n11_d2(self): self._check_case(11, 2)

    def test_even_n_raises(self):
        with self.assertRaises(ValueError):
            path_coord(0, 4, 2)

    def test_k_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            path_coord(27, 3, 3)  # 27 = n^d, out of range


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Hamiltonian Coverage (Theorem T2)
# ─────────────────────────────────────────────────────────────────────────────

class TestHamiltonianCoverage(unittest.TestCase):
    """
    THEOREM T2 (Hamiltonian Path), STATUS: PROVEN.

    The traversal visits every lattice point in Z_n^D exactly once.
    Verified by checking that the set of all generated coordinates has
    cardinality n^D (no duplicates, full coverage).
    """

    def _check_coverage(self, n: int, d: int) -> None:
        total      = n ** d
        coords_all = [path_coord(k, n, d) for k in range(total)]
        distinct   = len(set(coords_all))
        self.assertEqual(
            distinct, total,
            f"Coverage FAILED: n={n}, d={d}: "
            f"got {distinct} distinct coords, expected {total}"
        )

    def test_coverage_n3_d1(self):  self._check_coverage(3, 1)
    def test_coverage_n3_d2(self):  self._check_coverage(3, 2)
    def test_coverage_n3_d3(self):  self._check_coverage(3, 3)
    def test_coverage_n3_d4(self):  self._check_coverage(3, 4)
    def test_coverage_n5_d2(self):  self._check_coverage(5, 2)
    def test_coverage_n5_d3(self):  self._check_coverage(5, 3)
    def test_coverage_n7_d2(self):  self._check_coverage(7, 2)
    def test_coverage_n7_d3(self):  self._check_coverage(7, 3)
    def test_coverage_n11_d2(self): self._check_coverage(11, 2)

    def test_generate_path_array_bijection(self):
        """generate_path_array must produce each rank exactly once."""
        n, d  = 3, 3
        half  = n // 2
        arr   = generate_path_array(n, d)
        total = n ** d
        ranks = sorted(arr.flatten().tolist())
        self.assertEqual(ranks, list(range(total)),
                         "generate_path_array missing or duplicate ranks")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Latin Projection (Theorem T3)
# ─────────────────────────────────────────────────────────────────────────────

class TestLatinProjection(unittest.TestCase):
    """
    THEOREM T3 (Latin Hypercube Property), STATUS: PROVEN.

    Every axis-aligned 1-D projection of the traversal is a permutation
    of the signed digit set {-half, ..., +half}.
    """

    def _check_latin(self, n: int, d: int) -> None:
        half       = n // 2
        digit_set  = set(range(-half, half + 1))
        total      = n ** d
        coords_all = [path_coord(k, n, d) for k in range(total)]

        for axis in range(d):
            vals = set(c[axis] for c in coords_all)
            self.assertEqual(
                vals, digit_set,
                f"Latin FAILED: n={n}, d={d}, axis={axis}: "
                f"got {sorted(vals)}, expected {sorted(digit_set)}"
            )

    def test_latin_n3_d2(self):  self._check_latin(3, 2)
    def test_latin_n3_d3(self):  self._check_latin(3, 3)
    def test_latin_n3_d4(self):  self._check_latin(3, 4)
    def test_latin_n5_d2(self):  self._check_latin(5, 2)
    def test_latin_n5_d3(self):  self._check_latin(5, 3)
    def test_latin_n7_d2(self):  self._check_latin(7, 2)
    def test_latin_n7_d3(self):  self._check_latin(7, 3)
    def test_latin_n11_d2(self): self._check_latin(11, 2)

    def test_generate_path_array_latin(self):
        """
        In generate_path_array(n, d), every axis-aligned 1-D slice of step
        indices must contain n distinct values (Latin property via array).
        """
        n, d = 5, 2
        arr  = generate_path_array(n, d)
        for axis in range(d):
            for fixed_idx in range(n):
                slc = [fixed_idx if i != axis else slice(None) for i in range(d)]
                row = arr[tuple(slc)].tolist()
                self.assertEqual(
                    len(set(row)), n,
                    f"Array Latin FAILED: n={n}, d={d}, axis={axis}, idx={fixed_idx}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Step Bound (Theorem T4) [NEW IN V10]
# ─────────────────────────────────────────────────────────────────────────────

class TestStepBound(unittest.TestCase):
    """
    THEOREM T4 (Step Bound), STATUS: PROVEN.

    max torus-step = min(d, floor(n/2)).

    This is a V10 contribution: the audit's conjecture 'C <= 2' is
    refined to the exact formula min(d, floor(n/2)), proven here.
    """

    def _torus_step(self, a: int, b: int, n: int) -> int:
        diff = abs(b - a) % n
        return min(diff, n - diff)

    def _max_step(self, n: int, d: int) -> int:
        total = n ** d
        mx    = 0
        prev  = path_coord(0, n, d)
        for k in range(1, total):
            curr = path_coord(k, n, d)
            step = max(self._torus_step(curr[i], prev[i], n) for i in range(d))
            if step > mx:
                mx = step
            prev = curr
        return mx

    def test_step_bound_n3_d2(self):
        self.assertEqual(self._max_step(3, 2), min(2, 1),
                         "Step bound: n=3,d=2 should be 1")

    def test_step_bound_n3_d4(self):
        self.assertEqual(self._max_step(3, 4), min(4, 1),
                         "Step bound: n=3,d=4 should be 1")

    def test_step_bound_n5_d2(self):
        self.assertEqual(self._max_step(5, 2), min(2, 2),
                         "Step bound: n=5,d=2 should be 2")

    def test_step_bound_n5_d3(self):
        self.assertEqual(self._max_step(5, 3), min(3, 2),
                         "Step bound: n=5,d=3 should be 2")

    def test_step_bound_n7_d2(self):
        self.assertEqual(self._max_step(7, 2), min(2, 3),
                         "Step bound: n=7,d=2 should be 2")

    def test_step_bound_n7_d3(self):
        self.assertEqual(self._max_step(7, 3), min(3, 3),
                         "Step bound: n=7,d=3 should be 3")

    def test_step_bound_formula(self):
        """
        Verify the formula min(d, floor(n/2)) matches measurements
        for all cases in CASES.
        """
        for n, d in CASES:
            if n ** d > 5000:
                continue
            bound    = min(d, n // 2)
            measured = self._max_step(n, d)
            self.assertEqual(
                measured, bound,
                f"Step bound formula FAILED: n={n}, d={d}: "
                f"formula={bound}, measured={measured}"
            )

    def test_audit_conjecture_scope(self):
        """
        The audit conjectured 'C <= 2'.  This holds for a SUBSET of (n,d):
        n=3 (any d), d<=2 (any n), n=5 and d<=2.
        Verify the subset where the conjecture holds.
        """
        conjecture_cases = [(3, 2), (3, 3), (5, 2), (7, 2), (11, 2)]
        for n, d in conjecture_cases:
            measured = self._max_step(n, d)
            self.assertLessEqual(
                measured, 2,
                f"Conjecture C<=2 should hold for n={n},d={d}, got {measured}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Siamese D=2 (Theorem T5)
# ─────────────────────────────────────────────────────────────────────────────

class TestSieseD2(unittest.TestCase):
    """
    THEOREM T5 (Siamese Generalisation), STATUS: PROVEN.

    For D=2, the FM-Dance path reduces exactly to the classical
    Siamese (de la Loubere) magic-square algorithm.
    """

    def _check_siamese(self, n: int) -> None:
        result = verify_siamese_d2(n)
        self.assertTrue(
            result["bijection_ok"],
            f"Siamese n={n}: bijection failed"
        )
        self.assertTrue(
            result["primary_ok"],
            f"Siamese n={n}: primary step (-1,+1) mod n not satisfied. "
            f"Got steps: {result.get('primary_diffs', [])[:3]}"
        )
        self.assertTrue(
            result["siamese_ok"],
            f"Siamese n={n}: overall check failed"
        )

    def test_siamese_n3(self): self._check_siamese(3)
    def test_siamese_n5(self): self._check_siamese(5)
    def test_siamese_n7(self): self._check_siamese(7)
    def test_siamese_n11(self): self._check_siamese(11)

    def test_siamese_primary_step_is_diagonal(self):
        """
        For ALL n, the primary step of FM-Dance D=2 is the diagonal
        vector (-1, +1) in torus coordinates (equivalently (n-1, 1) unsigned).
        """
        for n in [3, 5, 7, 11]:
            d = 2
            total = n * n
            coords = [path_coord(k, n, d) for k in range(total)]
            for k in range(total - 1):
                if (k + 1) % n != 0:  # no carry
                    delta = tuple((coords[k+1][i]-coords[k][i]+n) % n for i in range(d))
                    self.assertEqual(
                        delta, (n - 1, 1),
                        f"Primary step wrong at n={n}, k={k}: got {delta}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Torus Cycle (Conjecture C4)
# ─────────────────────────────────────────────────────────────────────────────

class TestTorusCycle(unittest.TestCase):
    """
    CONJECTURE C4 (Torus Cycle Closure), STATUS: CONJECTURE.

    Tests whether the last point Phi(n^D - 1) and first point Phi(0) are
    within the step bound min(d, floor(n/2)) on the torus.

    This test is INFORMATIONAL — C4 is not yet proven analytically.
    If this test passes for all CASES, it supports the conjecture.
    """

    def _torus_dist(self, a: Tuple[int, ...], b: Tuple[int, ...], n: int) -> int:
        return max(min(abs(a[i]-b[i]) % n, n - abs(a[i]-b[i]) % n)
                   for i in range(len(a)))

    def _check_cycle(self, n: int, d: int) -> None:
        first = path_coord(0, n, d)
        last  = path_coord(n ** d - 1, n, d)
        dist  = self._torus_dist(first, last, n)
        bound = min(d, n // 2)
        self.assertLessEqual(
            dist, bound,
            f"Torus cycle FAILED: n={n}, d={d}: "
            f"first={first}, last={last}, dist={dist}, bound={bound}"
        )

    def test_torus_cycle_n3_d2(self):  self._check_cycle(3, 2)
    def test_torus_cycle_n3_d3(self):  self._check_cycle(3, 3)
    def test_torus_cycle_n3_d4(self):  self._check_cycle(3, 4)
    def test_torus_cycle_n5_d2(self):  self._check_cycle(5, 2)
    def test_torus_cycle_n5_d3(self):  self._check_cycle(5, 3)
    def test_torus_cycle_n7_d2(self):  self._check_cycle(7, 2)
    def test_torus_cycle_n7_d3(self):  self._check_cycle(7, 3)
    def test_torus_cycle_n11_d2(self): self._check_cycle(11, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Mean-Centering (PFNT-2)
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanCentered(unittest.TestCase):
    """
    THEOREM PFNT-2 (Mean-Centering), STATUS: PROVEN.

    For odd n and signed representation, the mean of all visited
    coordinates across each axis is exactly 0.
    """

    def _check_mean(self, n: int, d: int) -> None:
        coords = np.array([path_coord(k, n, d) for k in range(n ** d)], float)
        means  = coords.mean(axis=0)
        self.assertTrue(
            np.allclose(means, 0.0, atol=1e-10),
            f"Mean-centering FAILED: n={n}, d={d}, means={means}"
        )

    def test_mean_n3_d2(self):  self._check_mean(3, 2)
    def test_mean_n3_d3(self):  self._check_mean(3, 3)
    def test_mean_n5_d2(self):  self._check_mean(5, 2)
    def test_mean_n7_d3(self):  self._check_mean(7, 3)
    def test_mean_n11_d2(self): self._check_mean(11, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Fractal Low-Dim Independence (Theorem T6)
# ─────────────────────────────────────────────────────────────────────────────

class TestFractalLowDim(unittest.TestCase):
    """
    THEOREM T6 (Fractal Block Structure), STATUS: PROVEN.

    The low d_split coordinates of path_coord(k,n,d) depend only on the
    low-order d_split base-n digits of k.
    """

    def _check_fractal(self, n: int, d: int, split: int) -> None:
        result = verify_fractal(n, d, split)
        self.assertTrue(
            result["fractal_ok"],
            f"Fractal T6 FAILED: n={n}, d={d}, split={split}, "
            f"max_error={result['max_error']}"
        )

    def test_fractal_n3_d3_split1(self): self._check_fractal(3, 3, 1)
    def test_fractal_n3_d4_split2(self): self._check_fractal(3, 4, 2)
    def test_fractal_n5_d3_split1(self): self._check_fractal(5, 3, 1)
    def test_fractal_n5_d4_split2(self): self._check_fractal(5, 4, 2)
    def test_fractal_n7_d3_split1(self): self._check_fractal(7, 3, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — Theorem Registry Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestTheoremRegistry(unittest.TestCase):
    """
    Verify the theorem registry has all expected entries with correct statuses.
    """

    def test_proven_count(self):
        """Registry must contain at least 30 proven theorems (V12 sprint added BFRW-1,
        C3W, C3W-APN, C4, CGW, N-ARY-1, SA-1, SRM, TORUS_DIAM and more)."""
        count = len(proven_theorems())
        self.assertGreaterEqual(count, 30,
                         f"Expected >= 30 proven theorems in V12, got {count}")

    def test_open_conjecture_count(self):
        """Registry must contain exactly 2 open items after V15.3.1 (DN1 now PROVEN).
        V15.3.1: DN1 PROVEN; remaining open: OD-16, OD-17."""
        count = len(open_conjectures())
        self.assertEqual(count, 2,
                         f"Expected 2 open items (CONJECTURE) after V15.3.1, got {count}")

    def test_all_fm_dance_theorems_present(self):
        """All 6 FM-Dance theorems T1–T6 must be in registry and PROVEN."""
        for key in ["T1", "T2", "T3", "T4", "T5", "T6"]:
            t = get_theorem(key)
            self.assertIsNotNone(t, f"Theorem {key} missing from registry")
            self.assertTrue(t.is_proven(), f"Theorem {key} not PROVEN")

    def test_all_fm_dance_conjectures_present(self):
        """V13 registry state for legacy conjectures C2/C3/C4:
          - C2: DISPROVEN_SCOPED (computationally disproved for general arrays in V11)
          - C3: PROVEN in V13 (Cayley quasigroup reduction)
          - C4 (Torus Cycle Closure): PROVEN in V12 sprint via TORUS_DIAM
        C1 is PROVEN as L2 (Holographic Repair).
        T8 and FM-1 promoted to PROVEN in V13.
        T8b is the single remaining open conjecture.
        """
        # C2 is disproven-scoped — must exist but NOT be open
        c2 = get_theorem("C2")
        self.assertIsNotNone(c2, "C2 missing from registry")
        self.assertNotEqual(c2.status, "CONJECTURE",
            "C2 was computationally disproved (DISPROVEN_SCOPED) — should not be open conjecture")

        # C3 is now PROVEN (V13)
        c3 = get_theorem("C3")
        self.assertIsNotNone(c3, "C3 missing from registry")
        self.assertTrue(c3.is_proven(), "C3 should be PROVEN after V13 Cayley proof")

        # C4 was proven in V12 sprint via TORUS_DIAM
        c4 = get_theorem("C4")
        self.assertIsNotNone(c4, "C4 missing from registry")
        self.assertTrue(c4.is_proven(), "C4 (Torus Cycle Closure) should be PROVEN (V12 sprint)")

        # C1 must no longer exist as a standalone conjecture
        c1 = get_theorem("C1")
        self.assertIsNone(c1,
            "C1 (Holographic Repair) was upgraded to PROVEN as L2; "
            "should not exist as a separate CONJECTURE entry")

    def test_latin_theorems_present(self):
        """L1, L2, L3 must all be PROVEN in registry."""
        for key in ["L1", "L2", "L3"]:
            t = get_theorem(key)
            self.assertIsNotNone(t, f"Latin theorem {key} missing")
            self.assertTrue(t.is_proven(), f"Latin theorem {key} not PROVEN")

    def test_spectral_theorems_present(self):
        """S1, S2 must be PROVEN in registry."""
        for key in ["S1", "S2"]:
            t = get_theorem(key)
            self.assertIsNotNone(t, f"Spectral theorem {key} missing")
            self.assertTrue(t.is_proven(), f"Spectral theorem {key} not PROVEN")

    def test_holographic_repair_upgraded(self):
        """C1 (Holographic Repair) must be PROVEN, registered as L2."""
        l2 = get_theorem("L2")
        self.assertIsNotNone(l2, "L2 (Holographic Repair) must be in registry")
        self.assertTrue(l2.is_proven(), "L2 must be PROVEN (upgraded from C1)")
        self.assertIn("UPGRADED", l2.name,
                      "L2 should note it was upgraded from C1")

    def test_t4_new_in_v10(self):
        """T4 (Step Bound) was formalised in V10."""
        t4 = get_theorem("T4")
        self.assertIn("V10", t4.name, "T4 should be marked as NEW IN V10")

    def test_no_conjecture_marked_proven(self):
        """V13 conjecture audit: C3, T8, FM-1 are now PROVEN (V13);
        C4 is PROVEN (V12 sprint); C2 is DISPROVEN_SCOPED.
        T8b is the single remaining open conjecture."""
        # C3 is now PROVEN (V13 Cayley proof)
        c3 = get_theorem("C3")
        self.assertIsNotNone(c3, "C3 missing")
        self.assertEqual(c3.status, "PROVEN", "C3 should be PROVEN after V13")

        # C4 was legitimately proven in V12 sprint — verify it IS proven
        c4 = get_theorem("C4")
        self.assertIsNotNone(c4, "C4 missing")
        self.assertEqual(c4.status, "PROVEN",
                         "C4 (Torus Cycle Closure) must be PROVEN after V12 sprint")

        # C2 is disproven/scoped — not a conjecture anymore
        c2 = get_theorem("C2")
        self.assertIsNotNone(c2, "C2 missing")
        self.assertIn(c2.status, ("DISPROVEN_SCOPED", "DISPROVEN"),
                      f"C2 should be DISPROVEN_SCOPED, got {c2.status}")

        # T8b promoted to PROVEN in V13 final (Digit Carry / L_inf-Gray-1 theorem)
        t8b = get_theorem("T8b")
        self.assertIsNotNone(t8b, "T8b missing")
        self.assertTrue(t8b.is_proven(), "T8b should be PROVEN after V13 final")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — Addressing vs. Path: distinct objects
# ─────────────────────────────────────────────────────────────────────────────

class TestAddressingVsPath(unittest.TestCase):
    """
    Verify that fm_dance.py (addressing bijection) and fm_dance_path.py
    (kinetic traversal) produce DIFFERENT coordinate sequences.

    They are mathematically distinct objects:
      - Addressing: coord_i = digit_i - half  (per-digit shift, no mixing)
      - Kinetic:    x_i = prefix_sum(a_0..a_i) mod n - half  (cross-digit coupling)

    Both are valid bijections Z_n^D; they differ in traversal order and
    structural properties (only the kinetic path has the Siamese/T5 property).
    """

    def test_they_differ_n3_d2(self):
        n, d = 3, 2
        path_seq  = [path_coord(k, n, d) for k in range(n**d)]
        addr_seq  = [addressing_coords(k, n, d) for k in range(n**d)]
        # They should cover the same set but in different order
        self.assertEqual(set(path_seq), set(addr_seq),
                         "Both bijections must cover Z_n^D")
        self.assertNotEqual(path_seq, addr_seq,
                            "Addressing and kinetic traversals should differ in order")

    def test_they_differ_n5_d2(self):
        n, d = 5, 2
        path_seq  = [path_coord(k, n, d) for k in range(n**d)]
        addr_seq  = [addressing_coords(k, n, d) for k in range(n**d)]
        self.assertEqual(set(path_seq), set(addr_seq))
        self.assertNotEqual(path_seq, addr_seq)

    def test_addressing_is_simple_digit_shift(self):
        """
        Addressing bijection: coord_i = digit_i - half (no cross-digit mixing).
        This is NOT the Siamese pattern.
        """
        n, d = 3, 2
        half = n // 2
        for k in range(n ** d):
            c_addr  = addressing_coords(k, n, d)
            # For addressing: coord_i = (k // n^i) % n - half
            rem = k
            for i in range(d):
                expected = rem % n - half
                self.assertEqual(c_addr[i], expected,
                                 f"Addressing formula wrong at k={k}, i={i}")
                rem //= n

    def test_kinetic_has_siamese_property_addressing_does_not(self):
        """
        Only the kinetic traversal has the primary step (-1,+1) pattern
        (Siamese property T5).  The addressing bijection does NOT.
        """
        n, d = 3, 2
        total = n * n
        # Kinetic: every non-carry step is (-1, +1) mod n
        coords_path = [path_coord(k, n, d) for k in range(total)]
        for k in range(total - 1):
            if (k + 1) % n != 0:
                delta_path = tuple((coords_path[k+1][i]-coords_path[k][i]+n) % n
                                   for i in range(d))
                self.assertEqual(delta_path, (n-1, 1),  # (-1,+1) in unsigned
                                 f"Kinetic should have Siamese step at k={k}")

        # Addressing: check that primary step is NOT uniformly (-1,+1)
        coords_addr = [addressing_coords(k, n, d) for k in range(total)]
        siamese_count = 0
        for k in range(total - 1):
            if (k + 1) % n != 0:
                delta_addr = tuple((coords_addr[k+1][i]-coords_addr[k][i]+n) % n
                                   for i in range(d))
                if delta_addr == (n-1, 1):
                    siamese_count += 1
        # Addressing step at k=1: digit_0 goes 0->1, no other change -> step (1,0)
        # It should NOT satisfy the Siamese pattern for all non-carry steps
        non_carry_count = total - (total // n)
        self.assertLess(
            siamese_count, non_carry_count,
            "Addressing should NOT have Siamese pattern for all non-carry steps"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary runner
# ─────────────────────────────────────────────────────────────────────────────

def run_summary() -> None:
    """Run all tests and print a compact summary."""
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__(__name__))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)

    print("\n" + "=" * 60)
    print("FM-Dance V10 Property Test Summary")
    print("=" * 60)
    print(f"  Tests run   : {result.testsRun}")
    print(f"  Failures    : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print(f"  Status      : {'ALL PASS' if result.wasSuccessful() else 'FAILURES DETECTED'}")
    print()


if __name__ == "__main__":
    run_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — Distribution / Projection Uniformity (audit test_distribution)
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributionUniformity(unittest.TestCase):
    """
    THEOREM T3 (Latin Hypercube), PFNT-2 (Mean-Centering), STATUS: PROVEN.

    Audit item: test_distribution.
    Each coordinate value must appear exactly n^{d-1} times across all
    n^d lattice points — the projection uniformity property.

    This is the histogram version of the Latin property: instead of
    checking slices, we count global occurrences per digit per axis.
    """

    def _check_distribution(self, n: int, d: int) -> None:
        half           = n // 2
        expected_count = n ** (d - 1)
        total          = n ** d
        digit_set      = list(range(-half, half + 1))

        coords_all = [path_coord(k, n, d) for k in range(total)]

        for axis in range(d):
            counts = {v: 0 for v in digit_set}
            for c in coords_all:
                counts[c[axis]] += 1

            for v in digit_set:
                self.assertEqual(
                    counts[v], expected_count,
                    f"Distribution FAILED: n={n}, d={d}, axis={axis}, "
                    f"digit={v}: count={counts[v]}, expected={expected_count}"
                )

    def test_dist_n3_d2(self):  self._check_distribution(3, 2)
    def test_dist_n3_d3(self):  self._check_distribution(3, 3)
    def test_dist_n3_d4(self):  self._check_distribution(3, 4)
    def test_dist_n5_d2(self):  self._check_distribution(5, 2)
    def test_dist_n5_d3(self):  self._check_distribution(5, 3)
    def test_dist_n7_d2(self):  self._check_distribution(7, 2)
    def test_dist_n7_d3(self):  self._check_distribution(7, 3)
    def test_dist_n11_d2(self): self._check_distribution(11, 2)

    def test_equal_occurrence_count_formula(self):
        """Each digit appears exactly n^{d-1} times: verify the formula."""
        for n, d in [(3, 2), (5, 3), (7, 2)]:
            expected = n ** (d - 1)
            half     = n // 2
            counts   = {}
            for k in range(n ** d):
                c = path_coord(k, n, d)
                counts[c[0]] = counts.get(c[0], 0) + 1
            for v in range(-half, half + 1):
                self.assertEqual(counts.get(v, 0), expected,
                    f"Formula n^(d-1) wrong: n={n},d={d},digit={v}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 12 — Latin Theorems L1, L2, L3 (theory_latin.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestLatinTheorems(unittest.TestCase):
    """
    THEOREMS L1 (Constant Line Sum), L2 (Holographic Repair), L3 (Fault Tolerance).
    All STATUS: PROVEN.

    Tests run on the canonical value hyperprism: M[i_0,...] = (sum idx) mod n, signed.
    """

    def _value_hp(self, n: int, d: int) -> np.ndarray:
        return build_value_hyperprism(n, d, signed=True)

    def test_line_sum_zero_n3_d2(self):
        r = verify_constant_line_sum(self._value_hp(3, 2), 3)
        self.assertTrue(r["line_sum_ok"], f"L1 n=3,d=2: {r}")

    def test_line_sum_zero_n5_d3(self):
        r = verify_constant_line_sum(self._value_hp(5, 3), 5)
        self.assertTrue(r["line_sum_ok"], f"L1 n=5,d=3: {r}")

    def test_line_sum_zero_n7_d3(self):
        r = verify_constant_line_sum(self._value_hp(7, 3), 7)
        self.assertTrue(r["line_sum_ok"], f"L1 n=7,d=3: {r}")

    def test_line_sum_constant_formula(self):
        """line_sum_constant() must return 0 for odd signed n."""
        for n in [3, 5, 7, 11]:
            self.assertEqual(line_sum_constant(n, signed=True), 0,
                             f"line_sum_constant({n}) should be 0")

    def test_holographic_repair_n3_d2(self):
        r = verify_holographic_repair(self._value_hp(3, 2), 3)
        self.assertTrue(r["repair_ok"], f"L2 repair n=3,d=2: {r}")

    def test_holographic_repair_n5_d2(self):
        r = verify_holographic_repair(self._value_hp(5, 2), 5)
        self.assertTrue(r["repair_ok"], f"L2 repair n=5,d=2: {r}")

    def test_holographic_repair_n7_d3(self):
        r = verify_holographic_repair(self._value_hp(7, 3), 7)
        self.assertTrue(r["repair_ok"], f"L2 repair n=7,d=3: {r}")

    def test_holographic_repair_multi_axis_agrees(self):
        """All D independent axes must recover the same value (L3)."""
        r = verify_holographic_repair(self._value_hp(7, 3), 7)
        self.assertTrue(r["multi_axis_ok"],
                        "L3: multi-axis recovery must be consistent")

    def test_holographic_repair_n11_d2(self):
        r = verify_holographic_repair(self._value_hp(11, 2), 11)
        self.assertTrue(r["repair_ok"], f"L2 repair n=11,d=2: {r}")

    def test_all_latin_theorems_n3_d3(self):
        r = verify_all_latin_theorems(self._value_hp(3, 3), 3, verbose=False)
        self.assertTrue(r["all_ok"], f"L1+L2+L3 n=3,d=3: {r}")

    def test_all_latin_theorems_n5_d2(self):
        r = verify_all_latin_theorems(self._value_hp(5, 2), 5, verbose=False)
        self.assertTrue(r["all_ok"], f"L1+L2+L3 n=5,d=2: {r}")

    def test_holographic_repair_status_is_proven(self):
        """The repair function must report STATUS: PROVEN."""
        r = verify_holographic_repair(self._value_hp(3, 3), 3)
        self.assertEqual(r["status"], "PROVEN",
                         "verify_holographic_repair must report PROVEN")


# ─────────────────────────────────────────────────────────────────────────────
# Test 13 — Spectral Theorems S1, S2 (theory_spectral.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectralTheorems(unittest.TestCase):
    """
    THEOREMS S1 (DC Zeroing), S2 (Mixed-Frequency Flatness).
    Both STATUS: PROVEN (for value/communion hyperprisms).

    S3 (Axial Nullification) remains CONJECTURE and is NOT tested here.
    """

    def _communion_hp(self, n: int, d: int) -> np.ndarray:
        """Build communion (add) hyperprism."""
        import math
        perms = [factoradic_unrank(j % math.factorial(n), n, signed=True)
                 for j in range(d)]
        arr = np.zeros([n] * d, dtype=float)
        for idx in np.ndindex(*[n] * d):
            arr[idx] = float(sum(int(perms[j][idx[j]]) for j in range(d)))
        return arr

    def test_dc_zero_n3_d3(self):
        r = verify_dc_zero(self._communion_hp(3, 3))
        self.assertTrue(r["dc_zero"], f"S1 DC n=3,d=3: {r}")

    def test_dc_zero_n5_d2(self):
        r = verify_dc_zero(self._communion_hp(5, 2))
        self.assertTrue(r["dc_zero"], f"S1 DC n=5,d=2: {r}")

    def test_dc_zero_n7_d2(self):
        r = verify_dc_zero(self._communion_hp(7, 2))
        self.assertTrue(r["dc_zero"], f"S1 DC n=7,d=2: {r}")

    def test_mixed_flat_n3_d3(self):
        r = verify_spectral_flatness(self._communion_hp(3, 3), 3)
        self.assertTrue(r["mixed_flat"], f"S2 flatness n=3,d=3: {r}")

    def test_mixed_flat_n5_d2(self):
        r = verify_spectral_flatness(self._communion_hp(5, 2), 5)
        self.assertTrue(r["mixed_flat"], f"S2 flatness n=5,d=2: {r}")

    def test_mixed_flat_n7_d2(self):
        r = verify_spectral_flatness(self._communion_hp(7, 2), 7)
        self.assertTrue(r["mixed_flat"], f"S2 flatness n=7,d=2: {r}")

    def test_spectral_scope_rank_array_not_flat(self):
        """
        S2 does NOT apply to rank arrays (generate_path_array).
        Verify that rank arrays have non-flat mixed spectrum,
        confirming the scope restriction in S2.
        """
        n, d = 5, 2
        rank_arr = generate_path_array(n, d).astype(float)
        rank_arr -= rank_arr.mean()
        r = verify_spectral_flatness(rank_arr, n)
        # Rank array should NOT be flat (variance > 0)
        self.assertFalse(r["mixed_flat"],
            "Rank arrays should NOT satisfy S2 mixed flatness — "
            "confirms S2 scope is limited to communion arrays")

    def test_s1_s2_together_communion(self):
        """Both S1 and S2 must hold simultaneously for communion arrays."""
        for n, d in [(3, 3), (5, 2)]:
            arr = self._communion_hp(n, d)
            s1  = verify_dc_zero(arr)
            s2  = verify_spectral_flatness(arr, n)
            self.assertTrue(s1["dc_zero"], f"S1 failed n={n},d={d}")
            self.assertTrue(s2["mixed_flat"], f"S2 failed n={n},d={d}")
