"""
tests/test_theory/test_registry.py
=====================================
Theorem registry integrity, API namespace exports, version checks,
and conjecture accounting for the V15 release.

"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import unittest
import flu
from flu.theory.theorem_registry import (
    REGISTRY, get_theorem, proven_theorems, open_conjectures,
    disproven_negative_results, status_report,
)


# ── Registry integrity ────────────────────────────────────────────────────────

def test_registry_is_non_empty():
    assert len(REGISTRY) > 0

def test_registry_has_expected_core_keys():
    expected = {
        "T1","T2","T3","T4","T5","T6","T7","T8",
        "L1","L2","L3","L4","S1","S2","S2-Prime",
        "C3","C3W","C3W-STRONG","C4",
        "PFNT-1","PFNT-2","PFNT-3","PFNT-4","PFNT-5",
        "FM-1","BFRW-1","N-ARY-1","TORUS_DIAM",
    }
    missing = expected - set(REGISTRY.keys())
    assert not missing, f"Missing theorems: {missing}"

def test_all_records_have_name():
    for key, rec in REGISTRY.items():
        assert rec.name, f"{key} has no name"

def test_all_records_have_valid_status():
    valid = {"PROVEN", "CONJECTURE", "DISPROVEN_SCOPED", "OPEN", "PARTIAL", "RETIRED"}
    for key, rec in REGISTRY.items():
        assert rec.status in valid, f"{key} invalid status '{rec.status}'"

def test_all_proven_have_proof():
    for key, rec in REGISTRY.items():
        if rec.status == "PROVEN":
            assert rec.proof and len(rec.proof.strip()) > 0, f"{key} empty proof"

def test_all_conjectures_have_statement():
    for key, rec in REGISTRY.items():
        if rec.status == "CONJECTURE":
            assert rec.statement and len(rec.statement.strip()) > 0

def test_proof_tiers_are_valid():
    for r in REGISTRY.values():
        if r.status == "PROVEN" and r.proof_status is not None:
            assert isinstance(r.proof_status, str) and len(r.proof_status) > 0

def test_empirical_proofs_have_computational_language():
    for key, rec in REGISTRY.items():
        if rec.status == "PROVEN" and rec.proof_status == "empirical":
            assert any(w in rec.proof.lower()
                       for w in ["verif", "comput", "exhaustive", "search", "tested"]), \
                f"Empirical proof for {key} lacks computation language"


# ── Proven theorems ───────────────────────────────────────────────────────────

def test_proven_count_v15():
    """V15.3: 65 PROVEN theorems (DN2 + 4 sub-theorems added)."""
    assert len(proven_theorems()) == 65, \
        f"Expected 65 PROVEN, got {len(proven_theorems())}"
   
def test_core_theorems_are_proven():
    for key in ["T1","T2","T3","T4","T5","T6","T7","T8",
                "L1","L2","L3","S1","S2","C3","FM-1","BFRW-1"]:
        rec = get_theorem(key)
        assert rec is not None and rec.status == "PROVEN", f"{key} not PROVEN"

def test_pfnt_theorems_all_proven():
    for key in ["PFNT-1","PFNT-2","PFNT-3","PFNT-4","PFNT-5"]:
        assert get_theorem(key).status == "PROVEN"

def test_v13_proof_upgrades_proven():
    for key in ["C3W-STRONG","S2-GAUSS"]:
        rec = get_theorem(key)
        assert rec is not None and rec.status == "PROVEN", f"{key} not PROVEN"

def test_v15_new_theorems_proven():
    """V15 additions: HAD-1, TSP-1, CRYPTO-1, LEX-1, INT-1, GEN-1, INV-1, DISC-1."""
    for key in ["HAD-1","TSP-1","CRYPTO-1","LEX-1","INT-1","GEN-1","INV-1","DISC-1"]:
        rec = get_theorem(key)
        assert rec is not None and rec.status == "PROVEN", f"V15 theorem {key} not PROVEN"

def test_od33_registered_and_proven():
    t = get_theorem("OD-33")
    assert t is not None and t.status == "PROVEN"
    assert t.proof_status == "algebraic_trivial_via_bijection"

def test_even1_registered_and_proven():
    """EVEN-1 must be registered, PROVEN, and carry the correct proof tier."""
    t = get_theorem("EVEN-1")
    assert t is not None, "EVEN-1 not found in registry"
    assert t.status == "PROVEN", f"Expected PROVEN, got {t.status}"
    assert t.proof_status == "algebraic_and_computational", \
        f"Expected algebraic_and_computational, got {t.proof_status}"
    assert "Kronecker" in t.name or "Even" in t.name, \
        f"Unexpected theorem name: {t.name}"


# ── Open conjectures ──────────────────────────────────────────────────────────

def test_open_conjectures_count_v15():
    """V15.3: 3 open items (all CONJECTURE).
    DN2 now PROVEN (V15.3); DN1, OD-16, OD-17 remain open."""
    conjs = open_conjectures()
    assert len(conjs) == 3, \
        f"Expected 3 open items (CONJECTURE), got {len(conjs)}: {[c.name for c in conjs]}"

def test_open_conjectures_have_correct_status():
    for rec in open_conjectures():
        assert rec.status in ("CONJECTURE", "PARTIAL"), \
            f"open_conjectures() returned {rec.name} with unexpected status {rec.status!r}"

def test_all_conjectures_have_closure_path():
    """Every conjecture must document a closure path."""
    for c in open_conjectures():
        assert "CLOSURE" in c.proof.upper(), \
            f"Conjecture {c.name!r} missing CLOSURE PATH"

def test_od16_od17_are_open():
    for key in ["OD-16", "OD-17"]:
        rec = get_theorem(key)
        assert rec is not None and rec.status == "CONJECTURE"

def test_dn1_registered():
    t = get_theorem("DN1")
    assert t is not None and t.status == "CONJECTURE"

def test_dn1_references_od26_and_fm1():
    t = get_theorem("DN1")
    joined = " ".join(t.references)
    assert "OD-26" in joined
    assert "FM-1" in joined

def test_hil1_and_dec1_registered():
    hil = get_theorem("HIL-1")
    assert hil is not None and hil.status == "RETIRED", f"HIL-1 should be RETIRED (V15.1.3), got {hil.status}"
    dec = get_theorem("DEC-1")
    assert dec is not None and dec.status == "PROVEN", f"DEC-1 should be PROVEN (V15.1.2), got {dec.status}"


# ── Disproven / scoped ────────────────────────────────────────────────────────

def test_disproven_results_not_empty():
    assert len(disproven_negative_results()) >= 1

def test_c2_is_disproven_scoped():
    assert get_theorem("C2").status == "DISPROVEN_SCOPED"

def test_disproven_results_correct_status():
    for rec in disproven_negative_results():
        assert rec.status == "DISPROVEN_SCOPED"


# ── Registry totals ───────────────────────────────────────────────────────────

def test_registry_total_count_v15():
    """V15.3: 70 total theorems (DN2 + 4 sub-theorems added)."""
    assert len(REGISTRY) == 70, \
        f"Expected 70, got {len(REGISTRY)}"

# ── get_theorem helpers ───────────────────────────────────────────────────────

def test_get_theorem_returns_correct_record():
    t1 = get_theorem("T1")
    assert t1 is not None
    assert "T1" in t1.name or "Bijection" in t1.name

def test_get_theorem_unknown_returns_none():
    assert get_theorem("NONEXISTENT_XYZ") is None


# ── Status report ─────────────────────────────────────────────────────────────

def test_status_report_is_string():
    report = status_report()
    assert isinstance(report, str) and len(report) > 100

def test_status_report_contains_v15():
    assert "V15" in status_report()

def test_status_report_contains_proven():
    assert "PROVEN" in status_report()

def test_status_report_contains_conjecture():
    report = status_report()
    assert "CONJECTURE" in report or "OPEN" in report


# ── flu namespace exports ─────────────────────────────────────────────────────

def test_flu_version_is_15():
    assert flu.__version__.startswith("15."), f"Expected V15.x.x, got {flu.__version__}"

def test_flu_exports_status_report():
    assert callable(flu.status_report)

def test_flu_exports_open_conjectures():
    assert callable(flu.open_conjectures)

def test_flu_exports_proven_theorems():
    assert callable(flu.proven_theorems)

def test_flu_exports_registry():
    assert flu.REGISTRY is not None and len(flu.REGISTRY) > 0

def test_flu_namespace_traversal():
    assert hasattr(flu, "traversal") and callable(flu.traversal.traverse)

def test_flu_namespace_latin():
    assert hasattr(flu, "latin") and callable(flu.latin.generate)

def test_flu_namespace_seeds():
    assert hasattr(flu, "seeds") and flu.seeds.GOLDEN_SEEDS is not None

def test_flu_namespace_theory():
    assert hasattr(flu, "theory") and callable(flu.theory.status_report)

def test_flu_namespace_nary():
    assert hasattr(flu, "nary") and callable(flu.nary.generate)


# ── V12 legacy registry completeness ─────────────────────────────────────────

def test_registry_contains_v12_theorems():
    """All V12 sprint theorems present in registry."""
    for tid in ["T8", "FM-1", "BFRW-1", "SA-1", "N-ARY-1"]:
        t = get_theorem(tid)
        assert t is not None, f"V12 theorem {tid} missing"

def test_registry_proven_count_at_least_29():
    """Registry has at least 29 proven theorems (V12 baseline)."""
    assert len(proven_theorems()) >= 29

def test_registry_conjecture_count_le_7():
    """Registry has at most 5 open items (T9 PROVEN, DEC-1 PROVEN in V15.1.2)."""
    assert len(open_conjectures()) <= 5


# ── OD-33 computational verification ─────────────────────────────────────────

def test_od33_consecutive_blocks_form_net():
    """OD-33: every consecutive n^d block of FM-Dance is a (0,d,d)-net."""
    from flu.core.fm_dance_path import path_coord

    def kinetic_coords_full(k, n, d):
        half = n // 2
        digits = [(k // n**i) % n for i in range(d)]
        coords = []
        for i, a in enumerate(digits):
            if i == 0:
                coords.append((-a) % n - half)
            else:
                s_prev = sum(digits[:i])
                coords.append((s_prev + a) % n - half)
        return tuple(coords)

    for n, d in [(3, 2), (3, 3), (5, 2), (7, 2)]:
        nd     = n**d
        block0 = set(path_coord(r, n, d) for r in range(nd))
        assert len(block0) == nd, f"Block b=0 not full n={n},d={d}"
        for b in [1, 2]:
            block_b = set(kinetic_coords_full(b*nd + r, n, d) for r in range(nd))
            assert block_b == block0, \
                f"OD-33 block b={b} n={n},d={d} differs from block 0"


# ── UNIF-1 Spectral Unification (V15.1.4) ────────────────────────────────────

def test_unif1_in_registry():
    """UNIF-1 is registered as PROVEN with algebraic_and_computational tier."""
    u = get_theorem("UNIF-1")
    assert u is not None, "UNIF-1 not found in registry"
    assert u.status == "PROVEN"
    assert u.proof_status == "algebraic_and_computational"
    assert "sum-separable" in u.statement.lower()
    assert "mixed" in u.statement.lower()


def test_s2_pn_condition_lifted():
    """S2 no longer carries the erroneous 'PN only' restriction (lifted by UNIF-1)."""
    s2 = get_theorem("S2")
    assert s2 is not None
    # The erroneous condition must be gone
    for cond in s2.conditions:
        assert "PROVEN only when seeds are PN" not in cond, (
            "S2 still carries the erroneous PN-only condition; should be lifted by UNIF-1"
        )
    # Should reference UNIF-1 now
    assert any("UNIF-1" in r for r in s2.references), (
        "S2 should reference UNIF-1 as the strengthening theorem"
    )


def test_unif1_computational_vanishing():
    """UNIF-1 computational verification: mixed DFT components of sum-separable
    communion arrays vanish for identity seeds (and by UNIF-1, for any seeds)."""
    import numpy as np

    for n in [3, 5, 7]:
        for d in [2, 3]:
            # Build M[x] = sum_a phi_a(x_a) with phi_a = identity (signed digit)
            # This is the canonical sum-separable communion array.
            half = n // 2
            axes = [list(range(-half, n - half)) for _ in range(d)]  # signed digits
            # Construct the d-dimensional array via broadcasting
            shape = [n] * d
            arr = np.zeros(shape, dtype=float)
            for a in range(d):
                # add phi_a(x_a) to every slice along axis a
                idx = [None] * d
                idx[a] = slice(None)
                arr += np.array(axes[a], dtype=float)[tuple(idx)]
            fft = np.fft.fftn(arr)
            mags = np.abs(fft)
            mixed = []
            for idx in np.ndindex(*[n] * d):
                if sum(1 for x in idx if x != 0) >= 2:
                    mixed.append(float(mags[idx]))
            if mixed:
                assert max(mixed) < 1e-8, (
                    f"UNIF-1: mixed DFT not vanishing for n={n}, d={d}; "
                    f"max_mixed={max(mixed):.2e}"
                )


def test_registry_total_59_proven():
    """V15.3: exactly 65 PROVEN theorems (DN2 + 4 sub-theorems added)."""
    proved = proven_theorems()
    assert len(proved) == 65, f"Expected 65 PROVEN, got {len(proved)}"

def test_registry_total_65():
    """V15.3: total 70 entries in the registry (DN2 + 4 sub-theorems added)."""
    from flu.theory.theorem_registry import REGISTRY
    assert len(REGISTRY) == 70, f"Expected 70 total in V15.3, got {len(REGISTRY)}"

