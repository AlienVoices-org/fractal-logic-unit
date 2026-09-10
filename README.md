# FLU — Fractal Logic Unit

**Version:** 15.4.0 · **License:** MIT · **Python:** 3.10+  
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

The library carries a **self-verifying theorem registry** with over 100 theorem entries
all cross-linked to the code that tests them.  Every claim is tagged with a proof tier;
nothing is asserted without evidence.

---

## Quick Start

```python
Quick Start
-----------
from flu import FractalNet, ScarStore
from flu.container.sparse import (
    SparseCommunionManifold,
    SparseEvenManifold,
    SparseForeignFieldManifold
)
from flu.core.factoradic import get_golden_seeds, unrank_optimal_seed
from flu.applications import ExperimentalDesign, FLUInitializer
from flu.theory.theorem_registry import status_report

# 1. Quasi-Monte Carlo points — beats random by ~20% in L2-star discrepancy
net = FractalNet(n=3, d=4)
pts = net.generate(729)          # shape (729, 4), values in[0, 1)

# 2. Sparse Communion Manifold (Odd n) + Lazy Arithmetics
# Dimension D is derived from the length of the seed list
seeds = get_golden_seeds(n=3, d=64)
M1 = SparseCommunionManifold(n=3, seeds=seeds)
M2 = SparseCommunionManifold(n=3, seeds=seeds[::-1])

# ArithmeticMixin allows lazy operator calculus across 3^64 cells without RAM allocation
M_combined = (M1 + M2) * -1
val = M_combined.at_rank(123456789)  # Evaluated pointwise in O(D * depth) time

# 3. Sparse Even Manifold (Even n)
# Resolves the parity collapse using sum-mod Kronecker decomposition
M_even = SparseEvenManifold(n=4, d=64)
val_even = M_even.at_rank(123456789)

# 4. Sparse Foreign Field Manifold
# Projects an external field or sub-manifold into the sparse lattice coordinate system
M_foreign = SparseForeignFieldManifold(n=3, d=64, field_fn=lambda coords: sum(coords))

# 5. APN Golden Seed — proven δ=2 (Almost Perfect Nonlinear)
seed = unrank_optimal_seed(k=0, n=5)   # zero-compute for n ≡ 2 (mod 3)

# Theorem registry
print(status_report())
```

No external dependencies beyond NumPy.  All tests run without pytest and are part of github CI workflow:

```bash
python run_tests.py
FLU V15 Test Suite — 34 test files
========================================================================
  PASSED   932   FAILED    0   ERRORS    0   SKIPPED  112   TOTAL 1044   (3.4s)
========================================================================
  ✓  All 932 tests passed.
```
*Note on Test Counts:* Depending on your Python environment and the installation of optional dependencies (e.g., torch, jax, pandas, matplotlib), the total number of executed tests will vary. Skipped tests will be noted in the output. All core mathematical proofs run exclusively on the Python standard library and NumPy.

---

## Module Map

```
src/flu/
├── applications/
│   ├── codes.py             # Error-correcting codes
│   ├── design.py            # ExperimentalDesign (Latin Hypercube Sampling)
│   ├── hadamard.py          # Generate hadamard
│   ├── lighthouse.py        # SIMULATION_ONLY cryptographic beacon
│   ├── neural.py            # FLUInitializer, DynamicFLUNetwork
│   └── quantum.py           # SIMULATION_ONLY quantum primitives
├── container/
│   ├── communion.py         # ⊗_φ fusion operator
│   ├── contract.py          # Contraction utilities
│   ├── export.py            # PyTorch/JAX buffer export
│   ├── manifold.py          # Full CommunionManifold
│   └── sparse.py            # Sparse Manifold communion and arithmetics + ScarStore [HM-1]
├── core/
│   ├── even_n.py            # Even-n support
│   ├── factoradic.py        # Lehmer codes, Golden Seeds, APN search
│   ├── fm_dance.py          # T1-T6: bijection, Latin, Hamiltonian, step bound
│   ├── fm_dance_path.py     # Kinetic theorems, path utilities
│   ├── fractal_3_6.py       # FractalHyperCell_3_6
│   ├── fractal_net.py       # FractalNet — OD-27 digital net (NEW, V14 audit)
│   ├── hypercell.py         # FLUHyperCell
│   ├── lo_shu.py            # LoShuHyperCell, 72-perspective automorphisms
│   ├── lo_shu_sudoku.py     # LoShu-Sudoku-HyperCell (DN1, V15.3)
│   ├── n_ary.py             # N-ary generalisation
│   ├── operators.py         # Operators and hook for sparse arithmetics and custom operators
│   ├── parity_switcher.py   # Parity-switched Latin arrays
│   └── vhdl_gen.py          # VHDL hardware export (divider-free odometer)
├── interfaces/
│   ├── base.py              # base class interface facets
│   ├── cohomology.py        # CohomologyFacet — DEC-1 PROVEN
│   ├── crypto.py            # CryptoFacet — CRYPTO-1 PROVEN
│   ├── curves.py            # SpaceFillingCurveFacet — Open R&D HIL-1 follow up (DESIGN_INTENT V16)
│   ├── design.py            # DesignFacet — Latin Hypercube Experimental Design Bridge (DESIGN_INTENT V16)
│   ├── digital_net.py       # Digital Net Facets — FractalNetCorputFacet and FractalNetKineticFacet - OD-27 PROVEN
│   ├── genetic.py           # GeneticFacet — Permutation Seed Portability - GEN-1 PROVEN
│   ├── gray_code.py         # GrayCodeFacet — FM-Dance as n-ary Gray Code Generator - T8 PROVEN
│   ├── hadamard.py          # HadamardFacet — Sylvester-Hadamard Matrix Generator via Communion - HAD-1 PROVEN
│   ├── hilbert.py           # HilbertFacet — HIL-1 RETIRED (V15.1.3)
│   ├── integrity.py         # IntegrityFacet — Local Conservative-Law Auditor / Sonde - INT-1 PROVEN
│   ├── invariance.py        # InvarianceFacet — Structural Isomorphism Regression - INV-1 PROVEN
│   ├── lexicon.py           # LexiconFacet — Bijective n-ary Alphanumeric Mapping - LEX-1 PROVEN
│   └── neural.py            # NeuralFacet — Bias-Free Neural Weight Initialisation Bridge (DESIGN_INTENT V16)
├── theory/
│   ├── theorem_registry.py  # 59-entry self-verifying theorem registry
│   ├── theory.py            # PhasedFractalNumberTheory (PFNT axioms)
│   |── theory_communion_algebra.py
│   ├── theory_container.py  # Permutation lattice algebra
│   ├── theory_fm_dance.py   # T1-T9, DN1-DN2, HM-1, OD-16-PM, OD-17-PM
│   ├── theory_latin.py      # L1, L2, L3 Latin theorems
│   └── theory_spectral.py   # S1, S2, S2-Prime spectral theorems
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

### 4. Sparse Manifold & Pointwise Calculus
```python
from flu.container.sparse import SparseCommunionManifold, SparseEvenManifold
from flu.core.factoradic import get_golden_seeds

# Odd n (Communion) requires permutations (seeds)
seeds = get_golden_seeds(n=3, d=64)
M_odd = SparseCommunionManifold(n=3, seeds=seeds)   # models 3^64 ≈ 10^30 cells

# Even n (Sum-Mod) requires only dimension
M_even = SparseEvenManifold(n=4, d=64)              # models 4^64 cells

# Pointwise Calculus (OPER-1)
# Manifolds inherit ArithmeticMixin, allowing O(1) memory expression trees
M_calc = (M_odd * 2) + 5
val = M_calc.at_rank(123456789)                     # O(D) evaluation time

store = ScarStore(n=3, d=4)
store.learn((1, -1, 0, 0), 99.0)         # record anomaly
store.recall((1, -1, 0, 0))              # 99.0 (scar)
store.recall((0, 0, 0, 0))               # baseline value
store.compression_ratio()                # n^D / (D + |scars|)
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
python run_tests.py               # all tests, no pytest required
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
supported by our theorems and empirical benchmarks at scales up to 3^256.
