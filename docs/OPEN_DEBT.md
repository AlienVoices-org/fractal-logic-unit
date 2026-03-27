# FLU — Open Debt Registry

**Status as of V15.3.2 (2026-03-28). DN1 PROVEN. Registry: 69 PROVEN · 73 total · 2 open.**

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 OPEN | Active conjecture — no proof known |
| 🟡 PARTIAL | Empirical evidence only; algebraic proof incomplete |
| 🔵 RESEARCH | Long-horizon research direction; not a blocker |

---

## Active Items

| ID | Name | Status |
|----|------|--------|
| DOC-1 | Eternal documentation debt (Closure forbidden). Update README's, PAPER.md & THEOREMS.md etc. | 🟡 PARTIAL |
| OD-16 | Delta-Min Conjecture Z_19 (all bijections) | 🔴 OPEN |
| OD-17 | Delta-Min Conjecture Z_31 (all bijections) | 🔴 OPEN |
| OD-34 | Kinship Curve — Hamiltonian continuity under RotationHub | 🔴 OPEN |
| OD-35 | Distributive Law: (M₁⊕M₂)⊗M₃ = (M₁⊗M₃)⊕(M₂⊗M₃) | 🔵 RESEARCH |
| OD-36 | APN Asymptotics: sparsity of APN bijections as n→∞ | 🔵 RESEARCH |
| OPER-3 | Sparse Forward Differences — O(D) derivative operator on manifolds | 🟡 PARTIAL |

---

## Recently Closed (V15.2–V15.3.1)

| ID | Claim | Closed in | How |
|----|-------|-----------|-----|
| ✅ DN1 | LoShu Sudoku Digital Net | **V15.3.1** | lo_shu_sudoku.py |
| ✅ OD-19 | OD-19-LINEAR — Linear Magic Hyperprism Uniqueness | **V15.3** | PROOF_OD19_LINEAR.md |
| ✅ DN2 | APN-Scrambled Digital Net | **V15.3** | ETK + Walsh + Variance + ANOVA |
| ✅ DN2-ETK | Discrepancy constant C_APN(D) = C_classic·(B/√n)^D | **V15.3** | ETK inequality + H-balancing |
| ✅ DN2-WALSH | Walsh-native discrepancy bound via digit-weight decay | **V15.3** | Walsh |ŵ(k)| ≤ (B/√n)^{μ(k)} |
| ✅ DN2-VAR | Owen-class variance bound, gain independent of smoothness | **V15.3** | Walsh variance framework |
| ✅ DN2-ANOVA | High-order interaction suppression, effective dim reduced | **V15.3** | Sobol' ANOVA decomposition |
| ✅ EVEN-1 | Even-n Latin Hyperprism via Kronecker Decomposition | V15.2.1 | Three-part algebraic + 83 tests |
| ✅ OD-27 | FractalNetKinetic (t,mD,D)-net with t=m(D-1) | V15.2 | T-Rank Lemma + depth decoupling |
| ✅ UNIF-1 | Spectral Unification of Sum-Separable Arrays | V15.1.4 | DFT linearity + char. orthogonality |
| ✅ T9 | FM-Dance Digital Sequence (generator matrices C_m = T) | V15.1 | Discrete integral identity |
| ✅ DEC-1 | ScarStore = coset decomposition C⁰(Z_n^D;Z_n)/SCM | V15.1.2 | Künneth + HM-1 |

---

## 1. Documentation Debt

### DOC-1 — README's / PAPER.md / THEOREMS.md : Theorem tables may be incomplete compared to source registry

**Status:** 🟡 PARTIAL

The debt DOC-1 can never be completed. It is eternal task to prevent documentation drift and improve it. 
Specific Issues can be added here:

| V15.3.1 | DN1 | LoShu Sudoku Digital Net (PROVEN) (lo_shu_sudoku.py)
| V15.3+ | OD-19 T8b | Linear Magic Hyperprism Uniqueness (PROVEN)(SCOPED) (PROOF_OD19_Linear.md)

Section 3 theorem tables (3.1–3.7) were not updated to include V15.2/V15.3 additions.
Missing from the tables: DN2 (now PROVEN), DN2-ETK, DN2-WALSH, DN2-VAR, DN2-ANOVA,
UNIF-1, DISC-1, FMD-NET, OD-32-ITER, SRM, C4, T7, LEX-1, INT-1, GEN-1, INV-1.
Section 4.7 (ScarStore) still refers to HM-1 as "conjecture" in prose.
**Fix:** Extend Section 3 with sub-section 3.8 covering V14/V15/V15.3 additions;
update §4.6 (FractalNet) with Owen scrambling results and DN2 proof summary.

---

## 2. Mathematical Debt

### OD-16 — Delta-Min Conjecture for Z_19

**Status:** 🔴 OPEN

**Statement:** No APN bijection (δ=2) exists over Z_19. Formally: δ_min(Z_19) = 3
for all bijections f: Z_19 → Z_19.

**Evidence:**
- OD-16-PM (PROVEN): all bijective power maps have δ ≥ 4 (Hasse-Weil).
- V14 search: extended polynomial families (binomials, trinomials, Dickson) — best δ=4.
- Random search: 8,000,000 trials, best δ=3 (~3.2% rate). No δ=2 found.
- GOLDEN_SEEDS[19] contains 8 best-available δ=3 seeds (documented as non-APN).

**What the V15.3 audit established:** The δ=3 seeds for n=19 form a separate weaker
result (DN2-δ3 proposition) with B_max = 2.463√19. The core DN2 theorem explicitly
excludes n=19. The question of whether any APN bijection exists remains fully open.

**Closure path:** GPU batch DDT (50M+ trials); algebraic obstruction via representation
theory of S_19; character sum analysis of all bijection families over Z_19.

---

### OD-17 — Delta-Min Conjecture for Z_31

**Status:** 🔴 OPEN

**Statement:** δ_min(Z_31) = 3 for all bijections over Z_31.

**Evidence:**
- OD-17-PM (PROVEN): all bijective power maps have δ ≥ 4.
- Random search: 3,300,000 trials, best δ=3 (~3.1% rate).
- GOLDEN_SEEDS[31] contains 8 best-available δ=3 seeds.

**Closure path:** Same as OD-16.

---

### OD-34 — Hamiltonian Continuity of the Kinship Curve

**Status:** 🔴 OPEN

**Statement:** Applying hyperoctahedral group actions Ω_j at carry levels j (the
`RotationHub`) preserves the Hamiltonian property (T2).

**Closure path:** Prove Ω_j acts as a symmetry of the local n^j sub-block, ensuring
the exit coordinate of block B_k matches the entry of B_{k+1}.

---

## 3. Arithmetic & Calculus

### OPER-3 — Sparse Forward Differences (The Derivative)

**Status:** 🟡 PARTIAL

**Requirement:** Define `M.delta(axis)` as an O(D) operator returning field of
differences: Δ_j M[x] = M[x + e_j] − M[x].

**Closure path:** Implement as a specialized `SparseArithmeticManifold` node querying
two related coordinates in the operator tree.

---

## 4. Long-Horizon Research

### OD-35 — Distributive Law

**Statement (candidate):** (M₁ ⊕ M₂) ⊗ M₃ = (M₁ ⊗ M₃) ⊕ (M₂ ⊗ M₃).
Close path: `InvarianceFacet` entropy measurement on both sides.

### OD-36 — APN Asymptotics

**Statement (candidate):** Characterise the sparsity of APN bijections as n → ∞.
The empirical rate ~3.2% for n=19, n=31 (δ=3 best) suggests a density argument.
Connection to algebraic geometry over finite fields (Weil bounds, character sums).

### NEW-1 — Generator Matrix Parameterisation Theorem

**Statement (candidate):** Every linear digital sequence over Z_n^D can be expressed
as σ(M·a(k)) for M ∈ GL(d, Z_n) and bijection σ: Z_n → Z_n. Discrepancy class, Latin
property, and spectral behaviour fully determined by (M, σ).
Subsumes T9 (M=T), FMD-NET (M=I), DN2 (σ=APN), Faure (M=P^m).
**Status:** 🔵 RESEARCH.

### NEW-3 — Min-entropy Hamiltonian Latin Uniqueness (OD-19 corollary)

**Status:** ✅ CLOSED (within linear-digit scope). OD-19-LINEAR PROVEN V15.3+; Corollary 3 of PROOF_OD19_LINEAR.md closes NEW-3 for linear-digit bijections. Full generality (all Gray-1 Ham. paths) remains RESEARCH.

### NEW-4 — Modular Pascal Fractal Dimension

**Statement (candidate):** Hausdorff dimension of the FM-Dance "Sierpiński Strata"
equals the dimension of the Pascal triangle mod n.
**Status:** 🔵 RESEARCH. BPT provides the carry distribution; fractal geometry
connection to Pascal mod n is the gap.

---
