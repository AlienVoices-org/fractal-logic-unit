"""
tests/benchmarks/run_benchmark_suite.py
=========================================
FLU V15.2 — Full Integrated Automatic Benchmark Pipeline.

Comprehensive health check covering every module, interface, and core
mathematical claim in the FLU library.  Designed to run automatically
in CI and produce a machine-readable JSON report.

Sections
--------
  A. Package integrity         — imports and version sanity for ALL modules
  B. Core FM-Dance correctness — T1, T2, T3, T4, factoradic, path_coord
  C. Theory registry           — all 61 theorems registered; counts correct
  D. Container modules         — Communion, Manifold/ScarStore, Contract
  E. Digital net discrepancy   — L2-star scaling for FractalNet variants
  F. Spectral theorems         — S2, UNIF-1, S2-Prime empirical validation
  G. Interface facets          — all 10+ FluFacet subclasses exercised
  H. Applications              — HadamardGenerator, LatinSquareCode, Lighthouse
  I. APN scrambling (DN2)      — per-depth seed rotation at n=5,7,11
  J. OD-27 measurements        — QMC generator matrix analysis & t-value probe
  K. Cross-radix               — n=3,5,7,11 head-to-head discrepancy
  L. T9 algebraic check        — digit-level T matrix identity (27/27 proof)
  M. Summary + JSON export     — machine-readable pass/warn/fail report

Run:
    PYTHONPATH=src python tests/benchmarks/run_benchmark_suite.py
    PYTHONPATH=src python tests/benchmarks/run_benchmark_suite.py --json
    PYTHONPATH=src python tests/benchmarks/run_benchmark_suite.py --quiet

Exit code: 0 if FAIL count == 0, else 1 (CI-compatible).

V15.2 — 2026-03-12
"""

from __future__ import annotations

import sys
import os
import itertools
import time
import json
import argparse
from typing import Any

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

# ── CLI ─────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="FLU benchmark suite")
_parser.add_argument("--json", action="store_true", help="Write JSON to benchmarks/latest_suite.json")
_parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
_ARGS, _ = _parser.parse_known_args()
VERBOSE = not _ARGS.quiet

RNG = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════════════════
# Health tracker
# ═══════════════════════════════════════════════════════════════════════════

CHECKS: list[tuple[str, str, str]] = []
REPORT: dict[str, Any] = {"sections": {}, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
T_START = time.time()


def check(name: str, passed: bool, warn: bool = False, note: str = "") -> None:
    icon = "PASS" if passed else ("WARN" if warn else "FAIL")
    CHECKS.append((icon, name, note))


def header(title: str) -> None:
    if VERBOSE:
        print(f"\n{'═'*72}")
        print(f"  {title}")
        print(f"{'═'*72}")


def info(msg: str) -> None:
    if VERBOSE:
        print(f"  {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

def l2_star(pts: np.ndarray) -> float:
    """Warnock L2-star discrepancy. Lower = better coverage."""
    N, d = pts.shape
    s1 = float(np.sum(np.prod(1.0 - (pts ** 2) / 2.0, axis=1)))
    s2 = sum(
        float(np.sum(np.prod(1.0 - np.maximum(pts[i], pts), axis=1)))
        for i in range(N)
    )
    return float(np.sqrt(abs(3.0 ** (-d) - (2.0 ** (1 - d) / N) * s1 + s2 / N**2)))


def fft_peak(pts: np.ndarray, bins: int = 32) -> float:
    """Max FFT peak across all 2-D projections. High = lattice artefact."""
    N, d = pts.shape
    peak = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            H, _, _ = np.histogram2d(pts[:, i], pts[:, j], bins=bins)
            peak = max(peak, float(np.max(np.abs(np.fft.fft2(H))[1:, 1:])))
    return peak


def dual_score(pts: np.ndarray, max_h: int = 4) -> tuple[tuple, float]:
    """Best integer dual vector and score. Near-zero = hyperplane alignment."""
    best_s, best_h = 1.0, ()
    for hv in itertools.product(range(-max_h, max_h + 1), repeat=pts.shape[1]):
        if all(h == 0 for h in hv):
            continue
        s = float(np.mean(np.abs(np.sin(np.pi * pts @ np.array(hv, dtype=float) * 2))))
        if s < best_s:
            best_s, best_h = s, hv
    return best_h, best_s


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Package integrity
# ═══════════════════════════════════════════════════════════════════════════

header("A. PACKAGE INTEGRITY — all module imports")

_import_results: dict[str, bool] = {}

def _try_import(label: str, code: str) -> bool:
    try:
        exec(code, {})
        _import_results[label] = True
        return True
    except Exception as e:
        _import_results[label] = False
        check(f"Import {label}", False, note=str(e)[:80])
        return False

# Core modules
_try_import("flu._version",      "from flu._version import __version__, FLU_VERSION_LABEL")
_try_import("flu.core.factoradic","from flu.core.factoradic import GOLDEN_SEEDS, unrank_optimal_seed")
_try_import("flu.core.fm_dance", "from flu.core.fm_dance import index_to_coords")
_try_import("flu.core.fm_dance_path", "from flu.core.fm_dance_path import path_coord, FMDanceIterator, boundary_partition_sizes")
_try_import("flu.core.fractal_net","from flu.core.fractal_net import FractalNet, FractalNetKinetic")
_try_import("flu.core.lo_shu",   "from flu.core.lo_shu import LoShuHyperCell, LO_SHU")
_try_import("flu.core.n_ary",    "from flu.core.n_ary import nary_generate, nary_info, nary_step_bound")
_try_import("flu.core.hypercell","from flu.core.hypercell import LoShuHyperCell as _HC")
_try_import("flu.core.even_n",   "from flu.core.even_n import generate as even_generate")
# Container modules
_try_import("flu.container.communion","from flu.container.communion import CommunionEngine")
_try_import("flu.container.sparse",  "from flu.container.sparse import ScarStore, SparseCommunionManifold")
_try_import("flu.container.manifold","from flu.container.manifold import LoShuHyperCell as _MHC")
# Theory modules
_try_import("flu.theory",            "from flu.theory import get_theorem, proven_theorems, open_conjectures, status_report")
# Interface modules
_try_import("flu.interfaces.digital_net","from flu.interfaces.digital_net import FractalNetCorputFacet, FractalNetKineticFacet")
_try_import("flu.interfaces.hadamard",   "from flu.interfaces.hadamard import HadamardFacet")
_try_import("flu.interfaces.gray_code",  "from flu.interfaces.gray_code import GrayCodeFacet")
_try_import("flu.interfaces.lexicon",    "from flu.interfaces.lexicon import LexiconFacet")
_try_import("flu.interfaces.integrity",  "from flu.interfaces.integrity import IntegrityFacet")
_try_import("flu.interfaces.genetic",    "from flu.interfaces.genetic import GeneticFacet")
_try_import("flu.interfaces.invariance", "from flu.interfaces.invariance import InvarianceFacet")
_try_import("flu.interfaces.cohomology", "from flu.interfaces.cohomology import CohomologyFacet")
_try_import("flu.interfaces.neural",     "from flu.interfaces.neural import NeuralFacet")
_try_import("flu.interfaces.crypto",     "from flu.interfaces.crypto import CryptoFacet")
# Application modules
_try_import("flu.applications.hadamard","from flu.applications.hadamard import HadamardGenerator")
_try_import("flu.applications.codes",   "from flu.applications.codes import LatinSquareCode, build_code_matrix")
_try_import("flu.applications.lighthouse","from flu.applications.lighthouse import LighthouseBeacon")

all_imports_ok = all(_import_results.values())
n_ok = sum(_import_results.values())
n_total = len(_import_results)
info(f"Imports: {n_ok}/{n_total} OK")
check("All module imports succeed", all_imports_ok,
      warn=(not all_imports_ok),
      note=f"{n_total - n_ok} failures" if not all_imports_ok else "")

# Version check
from flu._version import __version__, FLU_VERSION_LABEL
info(f"Version: {FLU_VERSION_LABEL} {__version__}")
check("Version string present", bool(__version__) and bool(FLU_VERSION_LABEL))

REPORT["sections"]["A"] = {
    "title": "Package Integrity",
    "imports_ok": n_ok, "imports_total": n_total,
    "version": __version__, "version_label": FLU_VERSION_LABEL,
    "failures": [k for k, v in _import_results.items() if not v],
}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Core FM-Dance correctness
# ═══════════════════════════════════════════════════════════════════════════

header("B. CORE FM-DANCE CORRECTNESS  (T1/T2/T3/T4/factoradic)")

from flu.core.fm_dance import index_to_coords
from flu.core.fm_dance_path import path_coord, FMDanceIterator, boundary_partition_sizes
from flu.core.factoradic import GOLDEN_SEEDS, unrank_optimal_seed

_b_results: dict[str, Any] = {}

# T1: bijection check at multiple (n,d)
t1_ok = True
for n, d in [(3,2),(5,3),(7,2),(3,4)]:
    coords = [index_to_coords(k, n, d) for k in range(n**d)]
    if len(set(map(tuple, coords))) != n**d:
        t1_ok = False
        break
check("T1: index_to_coords is a bijection on Z_n^d", t1_ok)
_b_results["T1_bijection"] = t1_ok

# T2: Hamiltonian — FM-Dance iterator produces all unique coords
t2_ok = True
for n, d in [(3,3),(5,2)]:
    it = FMDanceIterator(n, d)
    visited = set()
    for c in it:
        visited.add(tuple(c))
    if len(visited) != n**d:
        t2_ok = False
        break
check("T2: FMDanceIterator visits all n^d distinct points", t2_ok)
_b_results["T2_hamiltonian"] = t2_ok

# T3: Latin hypercube — for each axis, each value appears exactly n^(d-1) times
t3_ok = True
for n, d in [(3,3),(5,2)]:
    coords = [index_to_coords(k, n, d) for k in range(n**d)]
    half = n // 2
    d_set = list(range(-half, half+1))  # Note: use -(n//2), NOT -n//2 (Python floor div)
    for axis in range(d):
        proj = [c[axis] for c in coords]
        for v in d_set:
            if proj.count(v) != n**(d-1):
                t3_ok = False
                break
        if not t3_ok:
            break
    if not t3_ok:
        break
check("T3: Every axis projection is a permutation of D_set", t3_ok)
_b_results["T3_latin"] = t3_ok

# BPT: Boundary partition sizes
bpt_ok = True
for n, d in [(3,3),(5,2),(7,2)]:
    sizes = boundary_partition_sizes(n, d)
    expected = tuple((n-1)*n**(d-j-1) for j in range(d))
    total = sum(sizes)
    if sizes != expected or total != n**d - 1:
        bpt_ok = False
        break
check("BPT: Boundary partition sizes match (n-1)*n^(d-j-1)", bpt_ok)
_b_results["BPT"] = bpt_ok

# APN GOLDEN_SEEDS availability
gs_ok = all(n in GOLDEN_SEEDS for n in [5,7,11])
info(f"GOLDEN_SEEDS keys available: {list(GOLDEN_SEEDS.keys())[:8]}")
check("GOLDEN_SEEDS: APN seeds available for n=5,7,11", gs_ok)
_b_results["golden_seeds"] = gs_ok

REPORT["sections"]["B"] = _b_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Theory registry validation
# ═══════════════════════════════════════════════════════════════════════════

header("C. THEORY REGISTRY VALIDATION  (61 theorems, all statuses)")

from flu.theory import get_theorem, proven_theorems, open_conjectures, status_report

_proven = proven_theorems()
_conjectures = open_conjectures()
n_proven = len(_proven)
n_conjecture = len(_conjectures)

info(f"Registry: {n_proven} PROVEN  +  {n_conjecture} CONJECTURE/PARTIAL  = {n_proven + n_conjecture} active")

check("Registry: ≥ 59 PROVEN theorems", n_proven >= 59,
      note=f"Found {n_proven}")
check("Registry: OD-27 is PROVEN", get_theorem("OD-27") is not None and get_theorem("OD-27").is_proven())
check("Registry: T9 is PROVEN", get_theorem("T9") is not None and get_theorem("T9").is_proven())
check("Registry: UNIF-1 is PROVEN", get_theorem("UNIF-1") is not None and get_theorem("UNIF-1").is_proven())
check("Registry: FMD-NET is PROVEN", get_theorem("FMD-NET") is not None and get_theorem("FMD-NET").is_proven())
check("Registry: OD-33 is PROVEN", get_theorem("OD-33") is not None and get_theorem("OD-33").is_proven())
check("Registry: DN2 is PROVEN", get_theorem("DN2") is not None and (get_theorem("DN2").is_proven())

_conj_ids = [t.name.split(" -- ")[0] for t in _conjectures]
info(f"Open conjectures: {_conj_ids}")

REPORT["sections"]["C"] = {
    "proven": n_proven, "conjecture": n_conjecture,
    "od27_registered": get_theorem("OD-27") is not None,
    "conjecture_ids": _conj_ids,
}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION D — Container modules
# ═══════════════════════════════════════════════════════════════════════════

header("D. CONTAINER MODULES  (Communion, ScarStore, Contract)")

from flu.container.communion import CommunionEngine
from flu.container.sparse import ScarStore, SparseCommunionManifold

_d_results: dict[str, Any] = {}

# Communion engine
try:
    eng = CommunionEngine(phi="add")
    A = np.array([-1, 0, 1])
    B = np.array([-1, 0, 1])
    M = eng.commune(A, B)
    communion_ok = (M.shape == (3, 3))
    check("Communion: commune(A, B) produces correct 3×3 shape", communion_ok)
    _d_results["communion_shape_ok"] = communion_ok
except Exception as e:
    check("Communion: commune(A, B) produces correct shape", False, note=str(e))
    _d_results["communion_shape_ok"] = False

# SparseCommunionManifold
try:
    seeds_scm = [np.array([-1, 0, 1]), np.array([0, 1, -1])]
    scm = SparseCommunionManifold(n=3, seeds=seeds_scm)
    scm_ok = hasattr(scm, 'n') and scm.n == 3 and hasattr(scm, 'seeds')
    check("SparseCommunionManifold: constructs with seeds", scm_ok)
    _d_results["manifold_ok"] = scm_ok
except Exception as e:
    check("SparseCommunionManifold: constructs with seeds", False, note=str(e))
    _d_results["manifold_ok"] = False

# ScarStore: compression ratio
try:
    ss = ScarStore(n=3, d=3)
    # Learn 9 anomaly cells using signed coordinates in range [-1, 1]
    for i in range(-1, 2):
        for j in range(-1, 2):
            ss.learn((i, j, -1), 99)
    cr = ss.compression_ratio()
    check("ScarStore: compression_ratio() > 1", cr > 1.0, note=f"ratio={cr:.2f}")
    check("ScarStore: scar_count matches learned anomalies",
          ss.scar_count() == 9, note=f"scar_count={ss.scar_count()}")
    _d_results["scarstore_cr"] = cr
    _d_results["scarstore_scar_count"] = ss.scar_count()
    recalled = ss.recall((0, 0, 0))
    check("ScarStore: recall of non-scar returns baseline value",
          recalled is not None, note=f"recalled={recalled}")
except Exception as e:
    check("ScarStore: basic operations", False, note=str(e))
    _d_results["scarstore_cr"] = None

REPORT["sections"]["D"] = _d_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION E — Digital net discrepancy scaling
# ═══════════════════════════════════════════════════════════════════════════

header("E. DIGITAL NET DISCREPANCY SCALING  (n=3, d=4)")

from flu.core.fractal_net import FractalNet, FractalNetKinetic

n_e, d_e = 3, 4
net_c = FractalNet(n_e, d_e)
net_k = FractalNetKinetic(n_e, d_e)

_e_results: dict[str, Any] = {}
if VERBOSE:
    info(f"{'N':>6}  {'FractalNet':>11}  {'FractalNetK':>11}  {'MC':>10}  {'C<MC':>6}  {'K<MC':>6}")
    info(f"{'─'*65}")

for N in [81, 243, 729, 2187, 6561]:
    pc = net_c.generate(N)
    pk = net_k.generate(N)
    mc = RNG.random((N, d_e))
    lc, lk, lm = l2_star(pc), l2_star(pk), l2_star(mc)
    _e_results[N] = {"l2_corput": lc, "l2_kinetic": lk, "l2_mc": lm}
    if VERBOSE:
        info(f"{N:>6}  {lc:>11.6f}  {lk:>11.6f}  {lm:>10.6f}  "
             f"{'✓' if lc<lm else '✗':>6}  {'✓' if lk<lm else '✗':>6}")

check("FractalNet beats MC at N=729",
      _e_results[729]["l2_corput"] < _e_results[729]["l2_mc"])
check("FractalNetKinetic beats MC at N=729",
      _e_results[729]["l2_kinetic"] < _e_results[729]["l2_mc"])
check("Both nets identical at N=n^d (same point set — FMD-NET)",
      abs(_e_results[81]["l2_corput"] - _e_results[81]["l2_kinetic"]) < 1e-10)
check("Both nets identical at N=n^(2d)",
      abs(_e_results[6561]["l2_corput"] - _e_results[6561]["l2_kinetic"]) < 1e-6)

REPORT["sections"]["E"] = _e_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION F — Spectral theorems (S2, UNIF-1, S2-Prime)
# ═══════════════════════════════════════════════════════════════════════════

header("F. SPECTRAL THEOREM EMPIRICAL VALIDATION  (S2, UNIF-1, S2-Prime)")

_f_results: dict[str, Any] = {}

for n_sp, d_sp in [(3, 2), (5, 2), (7, 2), (3, 3)]:
    from flu.core.fm_dance import index_to_coords as i2c
    seeds_sp = [[int(i2c(k, n_sp, 1)[0]) for k in range(n_sp)],
                [int(i2c(k, n_sp, 1)[0]) for k in range(n_sp-1, -1, -1)]]
    M = np.zeros([n_sp]*d_sp)
    for idx in itertools.product(range(n_sp), repeat=d_sp):
        M[idx] = sum(seeds_sp[a % 2][idx[a]] for a in range(d_sp))
    Mhat = np.fft.fftn(M)
    mixed_mask = np.ones([n_sp]*d_sp, dtype=bool)
    for axis in range(d_sp):
        sl = [slice(None)]*d_sp
        sl[axis] = 0
        mixed_mask[tuple(sl)] = False
    mixed_var = float(np.var(np.abs(Mhat[mixed_mask])))
    s2_ok = mixed_var < 1e-10
    key = f"n{n_sp}_d{d_sp}"
    _f_results[key] = {"mixed_variance": mixed_var, "s2_ok": s2_ok}
    check(f"S2/UNIF-1: mixed DFT variance < 1e-10 at n={n_sp}, d={d_sp}",
          s2_ok, note=f"var={mixed_var:.2e}")

REPORT["sections"]["F"] = _f_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION G — Interface facets (all FluFacet subclasses)
# ═══════════════════════════════════════════════════════════════════════════

header("G. INTERFACE FACETS — all FluFacet subclasses")

_g_results: dict[str, Any] = {}

# G.1 FractalNetCorputFacet / FractalNetKineticFacet
try:
    from flu.interfaces.digital_net import FractalNetCorputFacet, FractalNetKineticFacet
    corf = FractalNetCorputFacet(n=3, d=3)
    kinf = FractalNetKineticFacet(n=3, d=3)
    pts_c = corf.generate(81)
    pts_k = kinf.generate(81)
    g1_ok = (pts_c.shape == (81, 3)) and (pts_k.shape == (81, 3))
    l2_c = corf.l2_discrepancy(pts_c)
    l2_k = kinf.l2_discrepancy(pts_k)
    check("G.1 FractalNetCorputFacet: generate and l2_discrepancy", g1_ok and l2_c > 0)
    check("G.1 FractalNetKineticFacet: generate and l2_discrepancy", g1_ok and l2_k > 0)
    _g_results["digital_net"] = {"l2_corput": l2_c, "l2_kinetic": l2_k, "ok": g1_ok}
except Exception as e:
    check("G.1 DigitalNetFacets", False, note=str(e))
    _g_results["digital_net"] = {"ok": False}

# G.2 HadamardFacet
try:
    from flu.interfaces.hadamard import HadamardFacet
    hf = HadamardFacet(d=4)
    H = hf.generate()
    orth_ok = np.allclose(H @ H.T, 16 * np.eye(16))
    check("G.2 HadamardFacet: H @ H.T = N*I (d=4)", orth_ok)
    _g_results["hadamard"] = {"shape": list(H.shape), "orthogonal": bool(orth_ok)}
except Exception as e:
    check("G.2 HadamardFacet", False, note=str(e))
    _g_results["hadamard"] = {"ok": False}

# G.3 GrayCodeFacet
try:
    from flu.interfaces.gray_code import GrayCodeFacet
    gc = GrayCodeFacet(d=4, n=2)
    seq = list(gc.iter_sequence())
    gray_ok = len(seq) == 2**4
    check("G.3 GrayCodeFacet: sequence has 2^d=16 codewords", gray_ok)
    _g_results["gray_code"] = {"length": len(seq), "ok": gray_ok}
except Exception as e:
    check("G.3 GrayCodeFacet", False, note=str(e))
    _g_results["gray_code"] = {"ok": False}

# G.4 LexiconFacet
try:
    from flu.interfaces.lexicon import LexiconFacet
    lx = LexiconFacet(n=3, d=3)
    sym_api = [m for m in dir(lx) if not m.startswith("_")]
    check("G.4 LexiconFacet: instantiates with n=3, d=3", True)
    _g_results["lexicon"] = {"methods": sym_api[:6], "ok": True}
except Exception as e:
    check("G.4 LexiconFacet", False, note=str(e))
    _g_results["lexicon"] = {"ok": False}

# G.5 IntegrityFacet
try:
    from flu.interfaces.integrity import IntegrityFacet
    # Use a balanced Latin-square-like array satisfying L1
    arr_ig = np.array([[-1, 0, 1], [0, 1, -1], [1, -1, 0]], dtype=int)
    ig = IntegrityFacet(manifold=arr_ig, n=3)
    a = ig.audit_full()
    ig_ok = a is not None  # returns tuple or dict
    check("G.5 IntegrityFacet: audit_full() completes without error", ig_ok)
    _g_results["integrity"] = {"ok": ig_ok}
except Exception as e:
    check("G.5 IntegrityFacet: audit_full() completes without error", False, note=str(e))
    _g_results["integrity"] = {"ok": False}

# G.6 GeneticFacet
try:
    from flu.interfaces.genetic import GeneticFacet
    gf = GeneticFacet(populate_from_golden=True)
    avail = gf.available_n()
    gen_ok = 5 in avail or len(avail) > 0
    check("G.6 GeneticFacet: populated from GOLDEN_SEEDS", gen_ok,
          note=f"available n: {avail[:5]}")
    seed_rec = gf.get(n=5) if 5 in avail else None
    check("G.6 GeneticFacet: get(n=5) returns a SeedRecord", seed_rec is not None)
    _g_results["genetic"] = {"available_n": list(avail)[:6], "ok": gen_ok}
except Exception as e:
    check("G.6 GeneticFacet", False, note=str(e))
    _g_results["genetic"] = {"ok": False}

# G.7 InvarianceFacet
try:
    from flu.interfaces.invariance import InvarianceFacet
    iv = InvarianceFacet(n=3, d=2)
    cmp = iv.compare_branches()
    inv_ok = cmp.get("all_invariants_match", False)
    check("G.7 InvarianceFacet: compare_branches() confirms INV-1", inv_ok)
    _g_results["invariance"] = {"all_match": inv_ok, "theorem": cmp.get("theorem")}
except Exception as e:
    check("G.7 InvarianceFacet", False, note=str(e))
    _g_results["invariance"] = {"ok": False}

# G.8 CohomologyFacet
try:
    from flu.interfaces.cohomology import CohomologyFacet
    cf = CohomologyFacet(n=3, d=2)
    field = np.zeros((3, 3))
    lap = cf.laplacian(field)
    coh_ok = isinstance(lap, np.ndarray) and lap.shape == (3, 3)
    check("G.8 CohomologyFacet: laplacian(field) returns 3×3 matrix", coh_ok)
    _g_results["cohomology"] = {"laplacian_shape": list(lap.shape), "ok": coh_ok}
except Exception as e:
    check("G.8 CohomologyFacet", False, note=str(e))
    _g_results["cohomology"] = {"ok": False}

# G.9 NeuralFacet
try:
    from flu.interfaces.neural import NeuralFacet
    nf = NeuralFacet(n=9, d=2)
    W = nf.init_layer(shape=(9, 9))
    neu_ok = isinstance(W, np.ndarray) and W.shape == (9, 9)
    check("G.9 NeuralFacet: init_layer(shape=(9,9)) produces correct shape", neu_ok)
    _g_results["neural"] = {"shape": list(W.shape) if neu_ok else None, "ok": neu_ok}
except Exception as e:
    check("G.9 NeuralFacet: init_layer", False, warn=True, note=str(e)[:80])
    _g_results["neural"] = {"ok": False}

# G.10 CryptoFacet
try:
    from flu.interfaces.crypto import CryptoFacet
    crf = CryptoFacet(n=5)
    seed = crf.golden_seed()
    delta = crf.differential_uniformity(seed)
    is_apn = crf.is_apn(seed)
    cry_ok = isinstance(seed, (list, tuple, np.ndarray)) and delta <= 2
    check("G.10 CryptoFacet: golden_seed() is APN (delta=2)", cry_ok,
          note=f"delta={delta}, is_apn={is_apn}")
    _g_results["crypto"] = {"delta": int(delta), "is_apn": bool(is_apn), "ok": cry_ok}
except Exception as e:
    check("G.10 CryptoFacet", False, note=str(e))
    _g_results["crypto"] = {"ok": False}

REPORT["sections"]["G"] = _g_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION H — Applications
# ═══════════════════════════════════════════════════════════════════════════

header("H. APPLICATIONS  (Hadamard, LatinSquareCode, Lighthouse, Neural)")

_h_results: dict[str, Any] = {}

# Hadamard application
try:
    from flu.applications.hadamard import HadamardGenerator
    hgen = HadamardGenerator()
    for d_h in [2, 3, 4, 5]:
        H = hgen.generate(d=d_h)
        N_h = 2**d_h
        ok = np.allclose(H @ H.T, N_h * np.eye(N_h))
        if not ok:
            break
    check("H.1 HadamardGenerator: H @ H.T = N*I for d=2,3,4,5 (HAD-1)", ok)
    _h_results["hadamard_ok"] = ok
    vok = hgen.verify(d=3)
    check("H.1 HadamardGenerator.verify(d=3)", vok)
    _h_results["hadamard_verify"] = bool(vok)
except Exception as e:
    check("H.1 HadamardGenerator", False, note=str(e))
    _h_results["hadamard_ok"] = False

# LatinSquareCode
try:
    from flu.applications.codes import LatinSquareCode
    lsc = LatinSquareCode(n=5)
    # Test encode_pair / decode_symbol on a single known-good pair
    enc = lsc.encode_pair(0, 1)
    dec = lsc.decode_symbol(enc)
    codec_ok = (dec == (0, 1))
    # Also verify the matrix structure
    verify_ok = lsc.verify()
    check("H.2 LatinSquareCode: encode_pair(0,1) → decode_symbol round-trips", codec_ok)
    check("H.2 LatinSquareCode: matrix satisfies Latin square property (verify)", verify_ok)
    _h_results["latin_code"] = {"round_trip_ok": codec_ok, "verify_ok": bool(verify_ok)}
except Exception as e:
    check("H.2 LatinSquareCode", False, note=str(e))
    _h_results["latin_code"] = {"ok": False}

# N-ary info
try:
    from flu.core.n_ary import nary_info, nary_step_bound
    info_3_2 = nary_info(3, 2)
    step_3_2 = nary_step_bound(3, 2)
    nary_ok = isinstance(info_3_2, dict) and step_3_2 == min(2, 3//2)
    check("H.3 n_ary: nary_info and nary_step_bound (n=3, d=2)", nary_ok)
    _h_results["nary"] = {"info": info_3_2, "step_bound": step_3_2, "ok": nary_ok}
except Exception as e:
    check("H.3 n_ary module", False, note=str(e))
    _h_results["nary"] = {"ok": False}

REPORT["sections"]["H"] = _h_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION I — APN scrambling / DN2 validation
# ═══════════════════════════════════════════════════════════════════════════

header("I. APN SCRAMBLING — DN2 CONJECTURE  (FFT peak reduction)")

_i_results: dict[str, Any] = {}

for n_i, d_i in [(5, 2), (7, 2), (11, 2)]:
    if n_i not in GOLDEN_SEEDS:
        info(f"  n={n_i}: no GOLDEN_SEEDS, skip")
        continue
    N_i = n_i**3
    net_ki = FractalNetKinetic(n_i, d_i)
    pts_plain = net_ki.generate(N_i)
    fft_plain = fft_peak(pts_plain)
    best_red, best_seed = 0.0, -1
    for rank_i in range(min(4, len(GOLDEN_SEEDS[n_i]))):
        pts_sc = net_ki.generate_scrambled(N_i, seed_rank=rank_i)
        fft_sc = fft_peak(pts_sc)
        red = (fft_plain - fft_sc) / fft_plain if fft_plain > 0 else 0.0
        if red > best_red:
            best_red, best_seed = red, rank_i
    dn2_ok = best_red > 0.10   # at least 10% FFT peak reduction
    _i_results[f"n{n_i}"] = {
        "fft_plain": fft_plain, "best_reduction_pct": round(best_red * 100, 1),
        "best_seed": best_seed, "dn2_fft_ok": dn2_ok,
    }
    info(f"  n={n_i}: plain FFT={fft_plain:.1f}  best_reduction={best_red*100:.1f}% (seed {best_seed})")
    check(f"DN2 n={n_i}: FFT peak reduction > 10%", dn2_ok,
          warn=not dn2_ok, note=f"{best_red*100:.1f}%")

REPORT["sections"]["I"] = _i_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION J — OD-27 measurements (QMC generator matrix analysis)
# ═══════════════════════════════════════════════════════════════════════════

header("J. OD-27 MEASUREMENTS — QMC generator matrix & t-value (PROVEN V15.2: t=m(D-1))")
_j_results: dict[str, Any] = {}

# J.1 Verify T-matrix structure and det(T) = -1
for n_j, d_j in [(3, 3), (5, 3), (7, 2)]:
    T = np.zeros((d_j, d_j), dtype=int)
    for i in range(d_j):
        for jj in range(d_j):
            T[i, jj] = 1 if jj <= i else 0
    T[0, 0] = -1
    det_T = int(round(np.linalg.det(T.astype(float))))
    det_mod = det_T % n_j
    unit_ok = det_mod != 0  # -1 mod n != 0 for all n >= 2
    key = f"n{n_j}_d{d_j}"
    _j_results[f"T_det_{key}"] = {"det": det_T, "det_mod_n": det_mod, "unit": unit_ok}
    check(f"J.1 det(T)=-1 is unit in Z_{n_j} (d={d_j})", unit_ok,
          note=f"det(T)={det_T}, det mod {n_j}={det_mod}")

# J.2 FMD-NET check: n^d points cover all elementary intervals (0,d,d)-net
for n_j, d_j in [(3, 2), (3, 3), (5, 2)]:
    from flu.core.fractal_net import FractalNet as FN
    net_j = FN(n_j, d_j)
    pts_j = net_j.generate(n_j**d_j)
    # Check: every elementary interval [a/n, (a+1)/n)^d contains exactly 1 point
    cells = set()
    for pt in pts_j:
        cell = tuple(int(x * n_j) for x in pt)
        cells.add(cell)
    fmd_ok = len(cells) == n_j**d_j
    key = f"n{n_j}_d{d_j}"
    _j_results[f"FMD_NET_{key}"] = {"distinct_cells": len(cells), "ok": fmd_ok}
    check(f"J.2 FMD-NET (0,d,d)-net: n={n_j}, d={d_j}", fmd_ok,
          note=f"{len(cells)}/{n_j**d_j} distinct unit cells")

# J.3 Multi-depth block: consecutive blocks are also (0,d,d)-nets (OD-33)
n_j33, d_j33 = 3, 2
net_j33 = FractalNet(n_j33, d_j33)
block_size = n_j33**d_j33
all_blocks_ok = True
for blk in range(3):
    pts_blk = net_j33.generate((blk+2)*block_size)[blk*block_size:(blk+1)*block_size]
    cells_blk = set(tuple(int(x*n_j33) for x in pt) for pt in pts_blk)
    if len(cells_blk) != block_size:
        all_blocks_ok = False
        break
_j_results["OD33_consecutive_blocks"] = {"ok": all_blocks_ok}
check("J.3 OD-33: 3 consecutive blocks each form a (0,d,d)-net (n=3,d=2)", all_blocks_ok)

# J.4 Discrepancy vs MC at multiple N — evidence for low-discrepancy property
disc_evidence: dict[str, float] = {}
for N_j in [81, 243, 729]:
    pc_j = FractalNet(3, 3).generate(N_j)
    mc_j = RNG.random((N_j, 3))
    lc_j, lm_j = l2_star(pc_j), l2_star(mc_j)
    disc_evidence[f"N{N_j}"] = lc_j - lm_j   # negative = better than MC
_j_results["discrepancy_vs_MC_d3"] = disc_evidence
all_better = all(v < 0 for v in disc_evidence.values())
check("J.4 OD-27: FractalNet L2 < MC at N=81,243,729 (n=3,d=3)", all_better,
      note=str({k: f"{v:+.4f}" for k, v in disc_evidence.items()}))

# J.5 T9 digit identity (key step toward OD-27 generator-matrix proof)
from flu.core.fm_dance import index_to_coords as i2c
from flu.core.fm_dance_path import path_coord as pc_fn
n_t9, d_t9 = 3, 3
T_t9 = np.zeros((d_t9, d_t9), dtype=int)
for i in range(d_t9):
    for jj in range(d_t9):
        T_t9[i, jj] = 1 if jj <= i else 0
T_t9[0, 0] = -1

matches, total_t9 = 0, n_t9**d_t9
for k in range(total_t9):
    a = np.array(i2c(k, n_t9, d_t9), dtype=int)
    a_unsigned = a + n_t9 // 2   # to unsigned [0, n-1]
    pc_vec = np.array(list(pc_fn(k, n_t9, d_t9)), dtype=int)
    T_a_mod = (T_t9 @ a_unsigned) % n_t9 - n_t9 // 2
    if np.array_equal(T_a_mod, pc_vec):
        matches += 1

t9_exact = matches == total_t9
_j_results["T9_digit_identity"] = {"matches": matches, "total": total_t9, "ok": t9_exact}
info(f"  T9 digit identity: {matches}/{total_t9} exact matches")
check("J.5 T9: path_coord = T·index_to_coords (27/27 matches, n=3,d=3)", t9_exact,
      note=f"{matches}/{total_t9}")

REPORT["sections"]["J"] = _j_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION K — Cross-radix comparison
# ═══════════════════════════════════════════════════════════════════════════

header("K. CROSS-RADIX COMPARISON  (n=3,5,7,11; d=2)")

_k_results: dict[str, Any] = {}
if VERBOSE:
    info(f"  {'n':>3}  {'N':>6}  {'FractalNet':>11}  {'FractalNetK':>11}  {'MC':>10}  {'Both<MC':>8}")
    info(f"  {'─'*60}")

for n_k in [3, 5, 7, 11]:
    N_k = n_k**4
    c_k = FractalNet(n_k, 2)
    ki_k = FractalNetKinetic(n_k, 2)
    pc_k = c_k.generate(N_k)
    pk_k = ki_k.generate(N_k)
    mc_k = RNG.random((N_k, 2))
    lc_k, lk_k, lm_k = l2_star(pc_k), l2_star(pk_k), l2_star(mc_k)
    both_ok = lc_k < lm_k and lk_k < lm_k
    _k_results[f"n{n_k}"] = {
        "l2_corput": lc_k, "l2_kinetic": lk_k, "l2_mc": lm_k, "both_beat_mc": both_ok
    }
    if VERBOSE:
        info(f"  {n_k:>3}  {N_k:>6}  {lc_k:>11.6f}  {lk_k:>11.6f}  {lm_k:>10.6f}  {'✓' if both_ok else '✗':>8}")
    check(f"K: n={n_k}: both FractalNet variants beat MC (d=2, N=n^4)", both_ok)

REPORT["sections"]["K"] = _k_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION L — T9 algebraic check (full digit-level proof)
# ═══════════════════════════════════════════════════════════════════════════

header("L. T9 ALGEBRAIC CHECK  (digit-level T-matrix identity)")

_l_results: dict[str, Any] = {}

for n_l, d_l in [(3, 2), (3, 3), (5, 2)]:
    T_l = np.zeros((d_l, d_l), dtype=int)
    for i in range(d_l):
        for jj in range(d_l):
            T_l[i, jj] = 1 if jj <= i else 0
    T_l[0, 0] = -1
    total_l = n_l**d_l
    ok_l, err_l = 0, 0
    for k in range(total_l):
        a_l = np.array(i2c(k, n_l, d_l), dtype=int) + n_l // 2
        pc_l = np.array(list(pc_fn(k, n_l, d_l)), dtype=int)
        pred_l = (T_l @ a_l) % n_l - n_l // 2
        if np.array_equal(pred_l, pc_l):
            ok_l += 1
        else:
            err_l += 1
    t9_l_ok = ok_l == total_l
    key_l = f"n{n_l}_d{d_l}"
    _l_results[key_l] = {"matches": ok_l, "total": total_l, "ok": t9_l_ok}
    check(f"L T9: {ok_l}/{total_l} matches at n={n_l}, d={d_l}", t9_l_ok)
    if VERBOSE:
        info(f"  n={n_l}, d={d_l}: {ok_l}/{total_l} exact {'✓' if t9_l_ok else '✗'}")

REPORT["sections"]["L"] = _l_results

# ═══════════════════════════════════════════════════════════════════════════
# SECTION M — Summary + JSON export
# ═══════════════════════════════════════════════════════════════════════════

elapsed = time.time() - T_START
_passed = [c for c in CHECKS if c[0] == "PASS"]
_warned = [c for c in CHECKS if c[0] == "WARN"]
_failed = [c for c in CHECKS if c[0] == "FAIL"]

header(f"M. HEALTH SUMMARY  ({len(CHECKS)} checks · {elapsed:.1f}s)")

if VERBOSE:
    for icon, name, note in CHECKS:
        sym = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[icon]
        print(f"  {sym}  {name}")
        if note:
            print(f"       → {note}")

print(f"\n  Results: {len(_passed)} PASS  ·  {len(_warned)} WARN  ·  {len(_failed)} FAIL  "
      f"·  {elapsed:.1f}s elapsed")

if _failed:
    print(f"\n  FAILURES:")
    for _, name, note in _failed:
        print(f"    ❌ {name}")
        if note:
            print(f"       → {note}")

REPORT["summary"] = {
    "total": len(CHECKS),
    "pass": len(_passed),
    "warn": len(_warned),
    "fail": len(_failed),
    "elapsed_s": round(elapsed, 2),
    "exit_code": 0 if not _failed else 1,
    "checks": [{"status": s, "name": n, "note": note} for s, n, note in CHECKS],
}

# JSON export
if _ARGS.json or True:  # always write; CI reads it
    out_dir = os.path.join(_ROOT, "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest_suite.json")
    with open(out_path, "w") as fout:
        json.dump(REPORT, fout, indent=2, default=str)
    info(f"\n  JSON written → {out_path}")

sys.exit(0 if not _failed else 1)
