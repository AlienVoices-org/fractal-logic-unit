# FLU — V15 Roadmap (Current)

**Current version:** 15.2.0
**Baseline:** 681/763 tests passing · **58 PROVEN** · 5 CONJECTURE · 65 total theorems
**Document updated:** 2026-03-12

---

## Active Research Directions

### Near-Term (V16)

1. **T9 Algebraic Proof** (OD-27) — characterise FractalNet generator matrix; show
   multi-depth accumulation is a generalised lattice rule. Uses Niederreiter digital net theory.
2. **DN2 Carry Redesign** — Apply per-depth linear maps A_m ∈ GL(d,Z_n) to break
   hyperplane artefact. Target: dual_vector_score > 0.05.
3. **HIL-1 Hamiltonian Verification** — Prove tuned FM-Dance path is Hamiltonian for
   d=2,3; unlock the locality bound proof.
4. **DEC-1 Spectral Proof** — Compute Laplacian spectrum on Z_n^D; prove
   HolRep = Δ⁻¹ orthogonal to DC component.
5. **OD-5 Literature** — Search cryptographic literature for no-APN-permutation
   results for p ≡ 1 mod 3.
6. **OD-19 GL-Orbit Uniqueness** — T8b family uniqueness via triangular system analysis.

### Medium-Term

7. **Formal Paper** (`docs/PAPER.md`) — Full V15 write-up: interfaces package,
   51 proven theorems, benchmarks.
8. **Theorem tier upgrades** — HAD-1, TSP-1, CRYPTO-1 are `sketch` tier;
   add computational validation → `sketch+test`.
9. **mypy --strict compliance** — Full type annotation pass on `src/flu/`.

### Long-Term Research

10. **DN1 Propagation Lemma** — Depth-crossing analysis of Lo Shu prefix-sum
    digits (hard; requires digital net theory).
11. **Continuous ScarStore** — Extend ScarStore to [0,1)^d via FractalNet (OD-27 dependent).
12. **RotationHub Odd-N Space-Filling Curve** *(NEW — V15.1.3)* —
    `RotationHub` in `flu.interfaces.hilbert` applies hyperoctahedral rotations at FM-Dance
    carry levels. HIL-1 was retired because its binary (n=2) framing is incompatible with
    FM-Dance's odd-n requirement. The concept is worth revisiting under a corrected framing:
    - Define as "ternary/quinary carry-level rotation" (n=3, 5 primary cases).
    - First verify the Hamiltonian property: does `HilbertFacet(d=2, n=3).check_hamiltonian()`
      return True? If not, the tuned path is semantically broken regardless of locality.
    - If Hamiltonian: measure `locality_score()` vs plain FM-Dance at d=2, n=3 and d=2, n=5.
    - Only formulate a new conjecture if locality improvement is confirmed numerically.
    - Avoid "Hilbert curve" terminology; use "space-filling locality optimization" instead.

---

## V16 Design Proposals (from Synthesis-Review 2026-03-12)

These are implementation targets, not theorem claims. Full design sketches in `FLUSynthesis.txt`.

### Operator Pattern (FLUOperator / TMatrix / APNPermute)

Define atomic `FLUOperator` objects that carry a `theorem_id` and are composable:

```python
class TMatrix(FLUOperator):      # theorem_id = "T9"
    def __call__(self, x): ...   # vectorised prefix-sum mod n

class APNPermute(FLUOperator):   # theorem_id = "DN2"
    def __call__(self, x): ...   # seed[x] lookup
```

Pipeline then becomes: `P ∘ I` where I = TMatrix, P = APNPermute.
**Benefit:** symbolic pre-verification, O(D) oracle, lazy evaluation.

### SparseArithmeticManifold (OPER-1)

Extend `SparseCommunionManifold` with operator overloading so that manifold
arithmetic stays lazy:

```python
Dissonance = (M1 - M2) * 0.5   # no computation — just node construction
val = Dissonance[0, 1, -1, 0]  # O(D * depth) evaluation on demand
```

**Key design rule:** `+` / `-` preserves mean-zero (OPER-2, trivially PROVEN);
`*` / `/` transforms the Latin value distribution into a general field — document
clearly that the Latin property is **not preserved** across multiplication.

### SynthesisFacet

A `FluFacet` convenience wrapper exposing the three-stage pipeline:

```python
class SynthesisFacet(FluFacet):
    theorem_id = "NEW-1"
    def resolve(self, coord): return P(T(coord))   # O(D)
```

### CommunionAlgebra

Operator-composition engine for building nested pipelines without materialising
tensors. The compose() method is associative (PFNT-5); the result implements
`__getitem__` for O(D·depth) oracle access.

**Boundary condition check before implementing:** the current `UKMCContract`
enforces identity-hash integrity on frozen operators; any compose() API must
respect the contract's `__setattr__` freeze guard.
