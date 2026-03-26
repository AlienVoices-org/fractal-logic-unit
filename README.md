# FLU — Phased Fractal Number Theory / Universal Fractal Logic Unit

**Version:** 15.3.0 · **License:** MIT · **Python:** 3.10+  
**Authors:** Felix Mönnich & The Kinship Mesh Collective

---

## What is FLU?

FLU is a self-contained Python library for **n-ary Latin hyperstructures** —
deterministic, bijective, balanced combinatorial objects over the torus ℤₙᴰ.

The core primitive is the **FM-Dance bijection**: an O(D) prefix-sum transform
that maps any integer rank to a unique coordinate in ℤₙᴰ such that every
axis-aligned slice is a permutation of the digit set.  This single construction
underpins Latin hypercubes, Gray-code odometers, APN cryptographic seeds,
VHDL hardware counters, neural weight initialisers, experimental designs, and
a quasi-Monte Carlo digital net.

The library carries a **self-verifying theorem registry** of 70 entries
(65 PROVEN, 3 CONJECTURE, 1 DISPROVEN_SCOPED, 1 RETIRED), all cross-linked to the
code that tests them.  Every claim is tagged with a proof tier; nothing is
asserted without evidence.

---

## Quick Start

```python
from flu import (
    FractalNet,               # OD-27 quasi-Monte Carlo digital net
    SparseCommunionManifold,  # O(1)-memory infinite-scale oracle
    ScarStore,                # holographic sparse memory (OD-31 prototype)
)
from flu.core.factoradic import unrank_optimal_seed, GOLDEN_SEEDS
from flu.applications import ExperimentalDesign, FLUInitializer
from flu.theory.theorem_registry import status_report

# Quasi-Monte Carlo points — beats random by ~20% in L2-star discrepancy
net = FractalNet(n=3, d=4)
pts = net.generate(729)          # shape (729, 4), values in [0, 1)

# Holographic manifold — evaluates 3^64 universe in O(64) without storing it
M = SparseCommunionManifold(n=3, d=64)
val = M[(1, -1, 0, 1, ...)]     # coordinate lookup, O(D) time

# APN Golden Seed — proven δ=2 (Almost Perfect Nonlinear)
seed = unrank_optimal_seed(k=0, n=5)   # zero-compute for n ≡ 2 (mod 3)

# Theorem registry
print(status_report())
```

No external dependencies beyond NumPy.  All tests run without pytest:

```bash
python run_tests.py
# PASSED 1004  FAILED 9   ERRORS 9   SKIPPED 8    TOTAL 1029
```

---

## Module Map

```
src/flu/
├── core/
│   ├── fm_dance.py          # T1-T6: bijection, Latin, Hamiltonian, step bound
│   ├── fm_dance_path.py     # Kinetic theorems, path utilities
│   ├── factoradic.py        # Lehmer codes, Golden Seeds, APN search
│   ├── fractal_net.py       # FractalNet — OD-27 digital net (NEW, V14 audit)
│   ├── lo_shu.py            # LoShuHyperCell, 72-perspective automorphisms
│   ├── hypercell.py         # FLUHyperCell
│   ├── fractal_3_6.py       # FractalHyperCell_3_6
│   ├── n_ary.py             # N-ary generalisation
│   ├── parity_switcher.py   # Parity-switched Latin arrays
│   ├── vhdl_gen.py          # VHDL hardware export (divider-free odometer)
│   └── even_n.py            # Even-n support
├── container/
│   ├── manifold.py          # Full CommunionManifold
│   ├── sparse.py            # SparseCommunionManifold + ScarStore [HM-1]
│   ├── communion.py         # ⊗_φ fusion operator
│   ├── contract.py          # Contraction utilities
│   └── export.py            # PyTorch/JAX buffer export
├── interfaces/
│   ├── lexicon.py           # LexiconFacet — LEX-1 PROVEN
│   ├── integrity.py         # IntegrityFacet — INT-1 PROVEN
│   ├── genetic.py           # GeneticFacet — GEN-1 PROVEN
│   ├── invariance.py        # InvarianceFacet — INV-1 PROVEN
│   ├── hilbert.py           # HilbertFacet — HIL-1 RETIRED (V15.1.3)
│   ├── cohomology.py        # CohomologyFacet — DEC-1 PROVEN
│   ├── gray_code.py         # GrayCodeFacet — T8 PROVEN
│   ├── crypto.py            # CryptoFacet — CRYPTO-1 PROVEN
│   └── hadamard.py          # HadamardFacet — HAD-1 PROVEN
├── applications/
│   ├── design.py            # ExperimentalDesign (Latin Hypercube Sampling)
│   ├── neural.py            # FLUInitializer, DynamicFLUNetwork
│   ├── codes.py             # Error-correcting codes
│   ├── quantum.py           # SIMULATION_ONLY quantum primitives
│   └── lighthouse.py        # SIMULATION_ONLY cryptographic beacon
├── theory/
│   ├── theorem_registry.py  # 59-entry self-verifying theorem registry
│   ├── theory_fm_dance.py   # T1-T9, DN1-DN2, HM-1, OD-16-PM, OD-17-PM
│   ├── theory.py            # PhasedFractalNumberTheory (PFNT axioms)
│   ├── theory_latin.py      # L1, L2, L3 Latin theorems
│   ├── theory_spectral.py   # S1, S2, S2-Prime spectral theorems
│   ├── theory_container.py  # Permutation lattice algebra
│   └── theory_communion_algebra.py
└── utils/
    ├── benchmarks.py        # full_benchmark_report
    ├── math_helpers.py      # is_odd, factorial, digit helpers
    ├── verification.py      # verify_bijection, differential_uniformity
    └── viz.py               # Optional matplotlib visualisations
```

---

## Key Features

### 1. FM-Dance Bijection (Core)
O(D) bijection ℤ → ℤₙᴰ with guaranteed Latin and Hamiltonian properties.
Proven theorems T1–T8b cover bijection, Latin hypercube, step bound, Gray
bridge, carry cascade, and spectral flatness.

### 2. FractalNet — Quasi-Monte Carlo Digital Net (OD-27)
```python
net = FractalNet(n=3, d=4)
pts = net.generate(729)       # plain
pts = net.generate_scrambled(729)  # FLU-Owen APN-scrambled (DN2 PROVEN)
```
Beats standard Monte Carlo in L2-star discrepancy. Uses FLU-Owen scrambling
(independent APN permutations per (depth, dimension)), which gives 11-15×
better discrepancy constant and 125× lower integration variance vs standard
Owen scrambling at D=3. DN2 is now fully proven (V15.3).

### 3. APN Golden Seeds
```python
from flu.core.factoradic import GOLDEN_SEEDS, unrank_optimal_seed

# Pre-computed verified seeds for n = 3, 5, 7, 11, 13, 17, 23, 29
seed = unrank_optimal_seed(0, n=5)   # δ=2 (APN), zero-compute for n ≡ 2 mod 3

# All bijective power maps for p=19, p=31 have δ≥4 (algebraically proven)
# See docs/PROOF_APN_OBSTRUCTION.md
```

### 4. Sparse Holographic Manifold
```python
M = SparseCommunionManifold(n=3, d=64)   # models 3^64 ≈ 10^30 cells
val = M[(0, 1, -1, ...)]                  # O(D) evaluation, O(D) RAM

store = ScarStore(n=3, d=4)
store.learn((1, -1, 0, 0), 99.0)         # record anomaly
store.recall((1, -1, 0, 0))              # 99.0 (scar)
store.recall((0, 0, 0, 0))              # baseline value
store.compression_ratio()               # n^D / (D + |scars|)
```

### 5. Applications
```python
from flu.applications import ExperimentalDesign, FLUInitializer

# Perfect Latin Hypercube — O(D) deterministic, no stochastic collision-check
ed = ExperimentalDesign(n_levels=5, n_factors=3)

# Zero-bias neural weight initialisation (S1 + S2-GAUSS guarantees)
init = FLUInitializer(n=9)
weights = init.initialise(shape=(9, 9))   # mean=0, flat mixed-freq spectrum
```

### 6. VHDL Export
```python
from flu.core.vhdl_gen import generate_vhdl
vhdl = generate_vhdl(n=3, d=4)   # divider-free n-ary Gray-code counter
```

---

## Theorem Registry

```bash
python -c "from flu.theory.theorem_registry import status_report; print(status_report())"
```

Current state (V15.0.0):
- **65 PROVEN** — T1–T10, L1–L4, S1–S2-Gauss–S2-Prime, UNIF-1, C3/C3W/C3W-STRONG/C3W-APN/C4, SA-1, N-ARY-1, PFNT-1–5, FM-1, BFRW-1, TORUS_DIAM, OD-16-PM, OD-17-PM, OD-27, OD-32-ITER, OD-33, FMD-NET, DISC-1, HM-1, CGW/BPT/KIB/SRM/T7, HAD-1, TSP-1, CRYPTO-1, LEX-1, INT-1, GEN-1, INV-1, T9, T8b, DEC-1, EVEN-1, YM-1, GEN-0, **DN2, DN2-ETK, DN2-WALSH, DN2-VAR, DN2-ANOVA**
- **3 CONJECTURES** — DN1 (Lo Shu net), OD-16 (δ-min Z₁₉ all bijections), OD-17 (δ-min Z₃₁ all bijections)
- **1 RETIRED** — HIL-1 (self-contradictory primary case n=2; n=2 forbidden by implementation)
- **1 DISPROVEN** — C2 (axial DFT nullification, scoped)

---

## Open Debt Summary

| Item | Description | Status |
|------|-------------|--------|
| OD-3  | BigInt overhead d > 256 | 🟡 PARTIAL |
| OD-5  | APN seeds n=19, 31 | 🔴 OPEN |
| OD-16 | δ_min Z_19 (all bijections) | 🔴 OPEN |
| OD-17 | δ_min Z_31 (all bijections) | 🔴 OPEN |

Full detail: `docs/OPEN_DEBT.md`

---

## Running Tests

```bash
python run_tests.py               # all 277 tests, no pytest required
python tests/benchmarks/bench_qmc_rigor.py    # 7-test QMC rigor suite
python tests/benchmarks/bench_discrepancy.py  # discrepancy vs random/Sobol
python tests/benchmarks/bench_apn_hub.py      # APN Golden Seeds benchmark
```

---

## Docs

| File | Contents |
|------|----------|
| `docs/OPEN_DEBT.md` | All open conjectures, research directions, closures |
| `docs/AUDIT_NOTES.md` | V14 audit findings (OA, AG(4,3), automorphisms, discrepancy) |
| `docs/PROOF_APN_OBSTRUCTION.md` | Algebraic proof: no APN power map for p ≡ 1 (mod 3) |
| `CHANGELOG.md` | Full version history V10–V15 |
| `src/flu/theory/THEOREM_REGISTRY.json` | Machine-readable registry snapshot |

---

## Mathematical Background

FLU is grounded in:
- **Combinatorial Design Theory** — Latin hypercubes, orthogonal arrays (OA)
- **Finite Geometry** — AG(4,3), affine maps over GF(3)
- **Quasi-Monte Carlo Methods** — digital nets, rank-1 lattice rules, discrepancy
- **Cryptography** — APN functions, differential uniformity, S-box design
- **Algebraic Geometry** — Hasse-Weil bounds on algebraic curves over finite fields
- **Graph Theory** — Cayley graphs, expander-like spectral mixing

The core claim — that a single bijection can simultaneously satisfy Latin,
Hamiltonian, Gray-code, spectral flatness, and quasi-random properties — is
supported by 53 proven theorems and empirical benchmarks at scales up to 3^256.
