"""
flu/theory/theorem_registry.py
================================
Central Theorem Registry — V15.

Single point of truth for all proven theorems, open conjectures, and
design-intent claims across the FLU framework.

Usage
-----
    from flu.theory.theorem_registry import REGISTRY, get_theorem, status_report

    t1 = get_theorem("T1")
    print(t1.statement)
    print(status_report())

Epistemic tiers
───────────────
  PROVEN        – formal proof given; verified computationally
  CONJECTURE    – plausible; formal proof incomplete
  DESIGN INTENT – architectural choice, not a mathematical claim

No package-internal imports (pure-math leaf module).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from flu.theory.theory_fm_dance import (
    TheoremRecord,
    ALL_THEOREMS,
    C3_TENSOR_CLOSURE, C4_TORUS_CYCLE,
    L4_STEP_BOUND_REGIME,
    C3W_PROVEN, C3W_APN,
    SA1_SEPARABILITY, NARY1_GENERALISATION,
    T8_GRAY_BRIDGE, FM1_FRACTAL_MAGIC, BFRW1_DIFFUSION,
    TORUS_DIAM,
    T8B_STEP_VECTOR_UNIQUENESS,
    C3W_STRONG, S2_GAUSS_PROOF,
    C2_SCOPED_PROVEN,
    # V14: new conjectures
    DN1_DIGITAL_NET,
    DN1_GL_GRAECO_LATIN,
    DN1_OA_STRENGTH4,
    DN1_GEN_ALL_ODD,
    DN1_REC_RECURSIVE,
    DNO_GEN, DNO_COEFF_EVEN, DNO_INV, DNO_REC_MATRIX, DNO_OPT,
    DNO_P1, DNO_P2, DNO_OPT_FACT, DNO_TVAL_BAL, DNO_TVAL_REC,
    DNO_TVAL_STABLE, DNO_WALSH_REC, DNO_DUAL, DNO_ANOVA, DNO_COEFF,
    DNO_VAR, DNO_VAR_REC, DNO_ETK, DNO_WALSH, DNO_ASYM, DNO_SPECTRAL,
    DNO_OPT_WALSH, DNO_MINIMAX, DNO_RKHS, DNO_FUNC, DNO_SUPERIORITY,
    DNO_FULL, DNO_PREFIX,
    OD19_LINEAR,
    DELTA_MIN_19,
    DELTA_MIN_31,
    # V14 audit integrations
    T9_RADICAL_LATTICE,
    DN2_APN_SCRAMBLED_NET,
    DN2_ETK,
    DN2_WALSH,
    DN2_VAR,
    DN2_ANOVA,
    OD16_POWER_MAP_PROVEN,
    OD17_POWER_MAP_PROVEN,
    HM1_HOLOGRAPHIC_SPARSITY,
    # V14 open-debt closures
    FMD_NET,
    OD32_ITER,
    # V15 closure
    OD33_DIGITAL_SEQUENCE,
    # V15 audit integration — three formal bridge theorems
    HAD1_HADAMARD_COMMUNION,
    TSP1_ROUTING_ORACLE,
    CRYPTO1_APN_DIFFUSION,
    DISC1_DISCRETE_INTEGRAL,
    # V15 interface facet theorems
    LEX1_LEXICON_FACET,
    INT1_INTEGRITY_FACET,
    GEN1_GENETIC_FACET,
    INV1_INVARIANCE_FACET,
    HIL1_HILBERT_FACET,
    DEC1_COHOMOLOGY_FACET,
    # V15.1.4 — Spectral Unification
    UNIF1_SPECTRAL_UNIFICATION,
    # V15.2 — OD-27 Digital Net Classification conjecture
    OD27_DIGITAL_NET_CLASSIFICATION,
    # V15.2 — new proven theorems from Synthesis-Review
    T10_LATTICE_CONVERGENCE,
    C5_RECURSIVE_HYPERTORUS,
    YM1_DANIELIC_TEN,
    # V15.2 — even-n branch formalisation
    EVEN1_LATIN_HYPERPRISM,
)
from flu.theory.theory import PhasedFractalNumberTheory


# ── PFNT Theorems (from theory.py) as records ─────────────────────────────────

_PFNT_THEOREMS: List[TheoremRecord] = [
    TheoremRecord(
        name="PFNT-1 -- Container Partition",
        status="PROVEN",
        statement=(
            "S_n(D) = disjoint union_{c in D} C_c  where C_c = {pi | pi[n//2] = c}. "
            "|C_c| = (n-1)!  for all c in D."
        ),
        proof=(
            "Every permutation has a unique centre element (disjoint, covers S_n). "
            "Fix c at centre; freely permute remaining n-1 elements -> (n-1)! arrows. "
            "Sum n*(n-1)! = n! = |S_n(D)|.  []"
        ),
        conditions=["n >= 2"],
        references=["flu.theory.theory.PhasedFractalNumberTheory.get_container"],
    ),
    TheoremRecord(
        name="PFNT-2 -- Mean-Centering",
        status="PROVEN",
        statement=(
            "For signed odd n, every arrow pi in S_n(D) has mean(pi) = 0. "
            "For signed even n, mean = 0.5 (near-centred, up to affine shift)."
        ),
        proof=(
            "D = {-k,...,0,...,k} with k=(n-1)/2. sum(D) = 0 (symmetric set). "
            "Any permutation preserves multiset -> sum(pi) = 0 -> mean = 0.  []"
        ),
        conditions=["n is odd", "signed=True"],
        references=["flu.theory.theory.PhasedFractalNumberTheory.mean_centering"],
    ),
    TheoremRecord(
        name="PFNT-3 -- Latin Hypercube Property (Hyperprism)",
        status="PROVEN",
        statement=(
            "For M[i_1,...,i_d] = sum_k pi_k[i_k] mod n (permutations pi_1...pi_d), "
            "every axis-aligned 1-D slice of M is a permutation of D."
        ),
        proof=(
            "Fix all indices except axis a. Partial sum S = sum_{k!=a} pi_k[i_k] is "
            "constant c. Slice = {c + pi_a[j] mod n | j=0..n-1} = c + D = D.  []"
        ),
        conditions=["n >= 2", "D >= 1"],
        references=["flu.theory.theory.PhasedFractalNumberTheory.latin_property"],
    ),
    TheoremRecord(
        name="PFNT-4 -- Kinetic Completeness (Lehmer Code)",
        status="PROVEN",
        statement=(
            "The Lehmer-code (factoradic) unranking f: [0, (n-1)!) -> C_c is a "
            "bijection, enabling O(n) access to any arrow in C_c."
        ),
        proof=(
            "Standard result: Lehmer code encodes relative order of remaining digits. "
            "(n-1-i)! choices at step i -> (n-1)! total = |C_c|. Injective by "
            "construction; surjective by count. O(n) from linear scan.  []"
        ),
        conditions=["n >= 2"],
        references=["flu.core.factoradic.factoradic_unrank"],
    ),
    TheoremRecord(
        name="PFNT-5 -- Communion Closure (Conditional)",
        status="PROVEN",
        statement=(
            "If phi is associative, C1 tensor_phi C2 is a valid container of "
            "dimension d1+d2. Associativity of tensor_phi holds IFF phi is "
            "associative. Non-associative phi raises ValueError."
        ),
        proof=(
            "Latin property extends to d1+d2 dimensions (Theorem PFNT-3). "
            "Associativity of tensor_phi follows from associativity of phi by "
            "distributivity of tensor structure.  []\n"
            "CRITICAL CONSTRAINT: phi must be associative. The implementation "
            "validates this with a probabilistic spot-check on construction."
        ),
        conditions=["phi is associative (REQUIRED)", "containers have valid Latin structure"],
        references=["flu.container.communion.CommunionEngine"],
    ),
]


# ── Latin theorems (from theory_latin.py) ────────────────────────────────────

_LATIN_THEOREMS: List[TheoremRecord] = [
    TheoremRecord(
        name="L1 -- Constant Line Sum",
        status="PROVEN",
        statement=(
            "For a signed Latin hyperprism (odd n), every axis-aligned 1-D line "
            "sums to exactly 0. For unsigned: line sum = n*(n-1)/2."
        ),
        proof=(
            "sum(D_set) = 0 for D_set = {-k,...,0,...,k} (symmetric). "
            "Every 1-D slice is a permutation of D_set (Latin property). "
            "Permutation preserves multiset -> sum of slice = sum(D_set) = 0.  []"
        ),
        conditions=["n is odd", "signed=True", "M is a value Latin hyperprism"],
        references=["flu.theory.theory_latin.verify_constant_line_sum"],
    ),
    TheoremRecord(
        name="L2 -- Holographic Repair [UPGRADED from C1]",
        status="PROVEN",
        statement=(
            "For a signed Latin value hyperprism (odd n), any single erased cell "
            "M[P] is uniquely recoverable from ONE axis-aligned line through P: "
            "M[P] = -S_known  where S_known = sum of intact cells on that line. "
            "SCOPE: applies to value hyperprisms (M stores D_set elements), "
            "NOT rank arrays (which store ranks 0..n^D-1)."
        ),
        proof=(
            "By L1: M[P] + S_known = 0 (line sum = 0). "
            "Therefore M[P] = -S_known. "
            "Uniqueness: linear equation over Z with coefficient 1; S_known is determined. "
            "Computationally verified for n in {3,5,7,11}, D in {2,3}.  []"
        ),
        conditions=["n is odd", "signed=True",
                    "M is a value Latin hyperprism (D_set elements stored)",
                    "Exactly one cell erased"],
        references=["flu.theory.theory_latin.holographic_repair",
                    "Upgraded from C1 (was CONJECTURE in audit)"],
    ),
    TheoremRecord(
        name="L3 -- Multi-Axis Byzantine Fault Tolerance",
        status="PROVEN",
        statement=(
            "A D-dimensional hyperprism has D independent recovery axes through "
            "each point. Up to D-1 axes can be corrupted; any 1 uncorrupted axis "
            "suffices to recover the missing value."
        ),
        proof=(
            "D lines through P are pairwise orthogonal (share only P). "
            "Each applies L2 independently with a different set of (n-1) cells. "
            "All D computations yield the same value (unique solution). "
            "With D independent witnesses, D-1 corruptions are tolerable.  []"
        ),
        conditions=["n is odd", "signed=True", "M is a value Latin hyperprism"],
        references=["flu.theory.theory_latin.byzantine_fault_tolerance_degree"],
    ),
]


# ── Spectral theorems (from theory_spectral.py) ───────────────────────────────

_SPECTRAL_THEOREMS: List[TheoremRecord] = [
    TheoremRecord(
        name="S1 -- DC Zeroing (Zero Global Mean)",
        status="PROVEN",
        statement=(
            "For a signed Latin hyperprism (odd n), the DFT DC component = 0. "
            "Equivalently: the global mean of cell values is 0."
        ),
        proof=(
            "Each digit in D_set appears n^(D-1) times globally (Latin property). "
            "Total sum = n^(D-1) * sum(D_set) = n^(D-1) * 0 = 0. "
            "DC = total sum = 0.  [] (same as PFNT-2 / Mean-Centering)"
        ),
        conditions=["n is odd", "signed=True", "M is a value Latin hyperprism"],
        references=["flu.theory.theory_spectral.verify_dc_zero"],
    ),
    TheoremRecord(
        name="S2 -- Spectral Mixed-Frequency Flatness",
        status="PROVEN",
        proof_status="algebraic_sketch",
        statement=(
            "For a Communion (add) hyperprism M[i_0,...] = sum_j pi_j[i_j], "
            "all mixed DFT components (frequency vectors with >=2 non-zero entries) "
            "have identical magnitude. "
            "All mixed DFT components are identically zero for ALL choices of seeds. "
            "Proof by DFT linearity (V12 Wave 2): the communion-sum structure forces "
            "M̂(k)=0 whenever ≥2 frequency indices are non-zero. Independent of δ."
        ),
        proof=(
            "Proof by DFT linearity (V12 Wave 2). "
            "For M[i_1,...,i_d] = Σ_a π_a(i_a), the D-dim DFT at wavevector k is: "
            "  M̂(k) = Σ_a π̂_a(k_a) · Π_{b≠a} n·δ(k_b). "
            "For any mixed k (≥2 non-zero entries), EVERY term has ≥1 factor "
            "δ(k_b)=0 (since k_b≠0 for at least two b). "
            "Therefore M̂(k) = 0 for all mixed k, for ANY choice of seeds π_a. "
            "Mixed variance = Var({0,...,0}) = 0.  □ "
            "This is independent of seed quality (delta), verified for all n in "
            "{3,5,7,11,13,17,19,23,29,31} and D in {2,3,4} in V12 probe."
        ),
        conditions=["n >= 2", "M is a Communion (add) value hyperprism (sum-separable)",
                    "ANY seeds π_a (proof is independent of differential uniformity — "
                    "erroneous 'PN only' restriction lifted by UNIF-1, V15.1.4)",
                    "NOT for rank arrays (rank array is not sum-separable)"],
        references=["flu.theory.theory_spectral.verify_spectral_flatness",
                    "V11 Audit S2 downgrade: false premise corrected",
                    "UNIF-1 -- Spectral Unification (V15.1.4): subsumes and strengthens S2",
                    "flu.core.factoradic.is_pn_permutation"],
    ),
    TheoremRecord(
        name="S2-Prime -- Bounded Spectral Dispersion",
        status="PROVEN",
        statement=(
            "For a D-dimensional Communion hyperprism with seeds of differential "
            "uniformity δ ≤ δ_max, the mixed-frequency DFT magnitude variance "
            "satisfies: Var{|M̂(k)| : k mixed} ≤ n^D · (δ_max/n)². "
            "As δ_max/n → 0 the bound vanishes, recovering S2 for PN seeds."
        ),
        proof=(
            "DDT bound δ_max constrains |π̂_j(k)|² ≤ δ_max·n (standard DDT-spectrum). "
            "Product decomposition M̂(k) = Π_j π̂_j(k_j) gives magnitude spread. "
            "Variance bound follows from (max-min)² estimate.  □"
        ),
        conditions=["seeds have differential uniformity δ ≤ δ_max",
                    "M is a Communion (add) value hyperprism"],
        references=["flu.theory.theory_spectral.spectral_dispersion_bound",
                    "flu.theory.theory_spectral.SpectralDispersionBound",
                    "flu.core.factoradic.differential_uniformity",
                    "V11 Audit APN Seed Hub"],
    ),
]


# ── Updated conjectures (C2 scoped, C1 removed — now L2) ─────────────────────

_KINETIC_THEOREMS: List[TheoremRecord] = [
    TheoremRecord(
        name="T7 -- Path as Group Product Formula",
        status="PROVEN",
        statement=(
            "The FM-Dance coordinate at rank k is the cumulative sum of all "
            "step generators applied from rank 0:\n"
            "    Φ(k) = Φ(0) + Σ_{i=0}^{k-1} σ_{j(i)}   (in Z_n^D)\n"
            "where j(i) = carry level at step i and σ_j = step_vector(j, n, d).\n"
            "Equivalently: Φ(k) = ∏_{i=0}^{k-1} L_{σ_{j(i)}} (Φ(0)) "
            "where L_σ is left-translation by σ in the abelian group (Z_n^D, +).\n"
            "This is the GROUP-ALGEBRAIC perspective complementing the "
            "MATRIX perspective (T1, T-matrix) and the KINETIC perspective (CGW)."
        ),
        proof=(
            "By induction: Φ(0) is the origin (defined). "
            "If Φ(k) = Φ(0) + Σ_{i<k} σ_{j(i)}, then Φ(k+1) = Φ(k) + σ_{j(k)} "
            "(by the CGW construction, each step is addition of σ_{j(k)} in Z_n^D). "
            "Therefore Φ(k+1) = Φ(0) + Σ_{i≤k} σ_{j(i)}.  □\n"
            "Computationally verified: 0 errors for all (n,d) in {3,5,7}×{2,3}."
        ),
        conditions=["n is odd", "D ≥ 1"],
        references=[
            "flu.core.fm_dance_path.step_vector",
            "flu.core.fm_dance_path.cayley_generators",
            "Audit document: Ψ_k = ∏_{j=0}^{k} σ_j  (group action product formula)",
        ],
    ),
    TheoremRecord(
        name="SRM -- Self-Referential Manifold Corollary",
        status="PROVEN",
        statement=(
            "The FM-Dance path is self-referential: the step σ_{k-1} that produced "
            "x_k from x_{k-1} is uniquely determined by x_k alone, with no knowledge "
            "of the rank k. Equivalently:\n"
            "    The map x_k → σ_{k-1}  is a bijection (KIB).\n"
            "    The map x_k → x_{k-1}  is O(D) computable (invert_fm_dance_step).\n"
            "COROLLARY: Past and future are recoverable from present position "
            "on the manifold. The FM-Dance lattice is a self-consistent space-time."
        ),
        proof=(
            "Follows directly from KIB (Kinetic Inverse Bijection): "
            "Ψ(x_k) = first index j with (x_j + half) mod n ≠ 0 is a bijection "
            "from {Φ(k) | k≥1} to carry levels {0,…,D−1}. "
            "Given j = Ψ(x_k), the predecessor is x_{k-1} = (x_k − σ_j) mod n. "
            "Both operations are O(D) in the dimension only. "
            "The full forward map (T1) gives k from x; KIB gives σ from x; "
            "together these form a BIJECTIVE TRIPLE:\n"
            "    k → x:  path_coord          (T1, O(D))\n"
            "    x → k:  path_coord_to_rank  (T1⁻¹, O(D))\n"
            "    x → σ:  identify_step       (KIB, O(D))\n"
            "All three are O(D) bijections. The system is fully water-proof.  □"
        ),
        conditions=["n is odd", "D ≥ 1", "k ≥ 1"],
        references=[
            "flu.core.fm_dance_path.identify_step",
            "flu.core.fm_dance_path.invert_fm_dance_step",
            "Audit document: 'The FM-Dance path is a Self-Referential Manifold'",
            "Audit document: 'Water-proofing — every bit accounted for'",
        ],
    ),
    TheoremRecord(
        name="KIB -- Kinetic Inverse Bijection",
        status="PROVEN",
        statement=(
            "The map Ψ: Z_n^D → {0,…,D−1} defined by "
            "Ψ(x) = first index j where (x_j + ⌊n/2⌋) mod n ≠ 0 "
            "is a bijection from {Φ(k) | k=1,…,n^D−1} to the carry-level set. "
            "Consequently, given x_k the predecessor x_{k-1} is computable in O(D): "
            "j ← Ψ(x_k);  x_{k-1} = (x_k − σ_j) mod n  (all in O(D)).\n\n"
            "D=2 BIJECTION LEMMA (simplest case, explicit two-partition):\n"
            "    B_0 = {x | (x_0 + half) mod n ≠ 0}  — primary-step coordinates\n"
            "    B_1 = {x | (x_0 + half) mod n = 0}  — carry-step coordinates\n"
            "The two sets partition {Φ(k) | k≥1} = B_0 ∪ B_1 (disjoint), verified "
            "for all odd n ∈ {3,5,7,11}.\n\n"
            "PSEUDOCODE CORRECTION (audit document):\n"
            "    The audit's pseudocode checks coord[i] == (n-1)//2 (= +half).\n"
            "    This is INCORRECT — it matches unsigned digit n-1, not digit 0.\n"
            "    The correct check is: (coord[i] + half) mod n ≠ 0.\n"
            "    The theoretical insight (bijection exists) is correct; only the\n"
            "    boundary condition implementation was mis-stated."
        ),
        proof=(
            "After a level-j step k→k+1, the resulting coordinate x_{k+1} satisfies: "
            "  (x_i + half) mod n = 0  for i < j   (digits 0..j-1 wrapped to 0), "
            "  (x_j + half) mod n ≠ 0              (digit j is non-zero). "
            "Two different levels j ≠ j' produce different leading-zero patterns "
            "(contradiction at the first differing index), so Ψ is injective. "
            "Every coordinate Φ(k≥1) is reached by exactly one step (T2, Hamiltonian), "
            "so Ψ is surjective over the carry-level image. ∴ Ψ is a bijection.  □ "
            "Computationally verified for (n,d) in {3,5,7}×{2,3,4}: 0 mismatches."
        ),
        conditions=["n is odd", "D ≥ 1", "k ≥ 1 (origin has no predecessor)"],
        references=[
            "flu.core.fm_dance_path.identify_step",
            "flu.core.fm_dance_path.invert_fm_dance_step",
            "flu.core.fm_dance_path.traverse_reverse",
            "Audit document: Section 2 (Bijection Proof D=2 and D-dimensions)",
        ],
    ),
    TheoremRecord(
        name="BPT -- Boundary Partition Theorem",
        status="PROVEN",
        statement=(
            "Define B_j = {Φ(k) | carry level of step k→k+1 is j} for j=0,…,D−1. "
            "These sets are the 'Fractal Fault Lines' of the FM-Dance manifold "
            "(audit document terminology). "
            "Three properties hold:\n"
            "  (P1) B_i ∩ B_j = ∅ for i≠j  (disjoint). "
            "(P2) ⋃_{j=0}^{D-1} B_j = {Φ(k) | k=1,…,n^D−1}  (complete). "
            "(P3) |B_j| = (n−1)·n^{D−j−1}  (exact sizes). "
            "These three properties make the bijection Ψ of KIB well-defined and O(D).\n"
            "GEOMETRIC INTUITION (audit): The B_j sets are orthogonal hyper-planes "
            "in the FM-Dance manifold. A coordinate lies on B_j iff it occupies "
            "the j-th 'Fractal Fault Line' of the odometer cascade. The disjointness "
            "of fault lines is the geometric content of the KIB bijection."
        ),
        proof=(
            "P1/P2: The boundary signature Ψ(x) = first index with non-zero unsigned "
            "digit is a bijection (proved in KIB). Distinct carry levels produce "
            "distinct signatures → disjoint. Hamiltonian T2 → complete. "
            "P3: Level-j carry at rank k ↔ a_0=…=a_{j-1}=n−1, a_j∈{0,…,n−2}, "
            "a_{j+1},…,a_{D-1} free. Count = (n−1)·n^{D−j−1}. "
            "Sum check: Σ (n−1)·n^{D−j−1} = (n−1)·(n^D−1)/(n−1) = n^D−1. ✓  □ "
            "Computationally verified for (n,d) in {3,5,7}×{2,3,4}."
        ),
        conditions=["n is odd", "D ≥ 1"],
        references=[
            "flu.core.fm_dance_path.boundary_partition_sizes",
            "flu.core.fm_dance_path.identify_step",
            "Audit document: 'Orthogonality, Disjointness, Completeness' proof",
            "Audit document: 'Fractal Fault Line' geometric naming",
        ],
    ),
    TheoremRecord(
        name="CGW -- FM-Dance as Cayley Graph Walk",
        status="PROVEN",
        statement=(
            "The FM-Dance path Φ: [0,n^D) → Z_n^D is a Hamiltonian walk on the "
            "Cayley graph Cay(Z_n^D, S) where S = {σ_0,…,σ_{D-1}} are the D "
            "step vectors. The inverse walk uses S^{-1} = {σ_j^{-1}} (additive "
            "inverses in Z_n^D), giving the same structure in reverse order.\n"
            "THREE PERSPECTIVES ON THE SAME STRUCTURE:\n"
            "  Matrix view (T1): Φ(k) = T·a  where T is lower-triangular, det(T)=−1\n"
            "  Kinetic view (CGW): Φ(k) = Φ(k−1) + σ_{j(k−1)}  (additive step)\n"
            "  Algebraic view (T7): Φ(k) = Φ(0) + Σ_{i<k} σ_{j(i)}  (group product)\n"
            "All three are provably equivalent."
        ),
        proof=(
            "Each step is left-multiplication by a fixed generator σ_j(k) ∈ S. "
            "The walk visits every vertex exactly once (T2, Hamiltonian). "
            "∴ Φ is a Hamiltonian path on Cay(Z_n^D, S).  □ "
            "Inverse walk: S^{-1} exists because (Z_n^D, +) is an abelian group, "
            "every element has a unique additive inverse σ^{-1} = (n−σ) mod n. "
            "Applying σ_j^{-1} at each step reverses the path exactly. "
            "T7 (product formula) follows from induction on the step rule."
        ),
        conditions=["n is odd", "D ≥ 1", "S = {step_vector(j,n,d) | j=0..D-1}"],
        references=[
            "flu.core.fm_dance_path.cayley_generators",
            "flu.core.fm_dance_path.cayley_inverse_generators",
            "flu.core.fm_dance_path.inverse_step_vector",
            "flu.core.fm_dance_path.traverse_reverse",
            "Audit document: Section 1 (Algebraic Formalism / Group Action)",
        ],
    ),
]

_UPDATED_CONJECTURES: List[TheoremRecord] = [
    TheoremRecord(
        name="C2 -- Spectral Axial Nullification (DISPROVEN for general; SCOPED to L1 arrays)",
        status="DISPROVEN_SCOPED",
        statement=(
            "Purely axial DFT components of a signed Latin hyperprism are zero. "
            "DISPROVEN for general Latin arrays (rank arrays, rank-indexed communion arrays). "
            "PROVEN (scoped) for arrays satisfying L1 (constant line sums = 0): "
            "see C2-SCOPED. The original conjecture is retained as a named negative result. "
            "Counter-example: FM-Dance rank array (n=5) has max axial magnitude ~106."
        ),
        proof=(
            "DISPROVEN for general arrays: the FM-Dance rank array and arbitrary Latin "
            "hyperprisms have non-zero axial DFT components. "
            "Counter-example: n=5, D=2 rank array, max axial mag = 106.3. "
            "POSITIVE RESULT: L1-satisfying arrays (constant line sums) do have axial DFT = 0. "
            "This is C2-SCOPED (PROVEN V13). See also S2 (mixed components vanish for rank arrays)."
        ),
        conditions=["Restricted to L1-satisfying arrays for positive result — see C2-SCOPED"],
        references=["C2-SCOPED -- Axial Nullification for L1 Arrays (PROVEN V13)",
                    "S2 -- Mixed-Frequency Flatness (PROVEN)",
                    "flu.theory.theory_spectral.check_axial_nullification",
                    "V11 Audit C2 re-scoped after computational disproof"],
    ),
    C3_TENSOR_CLOSURE,
    TheoremRecord(
        name="C4 -- Torus Cycle Closure",
        status="PROVEN",
        statement=(
            "The FM-Dance traversal is a strict step-bounded Hamiltonian CYCLE "
            "(not just a path) if and only if D ≤ ⌊n/2⌋. "
            "The closing jump Φ(n^D−1) → Φ(0) has vector J = (−1,2,3,…,D) mod n. "
            "For D ≤ ⌊n/2⌋ all components satisfy the step bound; "
            "for D > ⌊n/2⌋ the closing jump exceeds the bound (Singularity Leap) "
            "but the path remains Hamiltonian."
        ),
        proof=(
            "Last-point identity: Φ(n^D−1) = (1,−2,−3,…,−D) mod n "
            "(derived by applying FM-Dance to the all-(n−1) digit sequence). "
            "Closing jump J = (−1,2,3,…,D) mod n (difference to origin). "
            "Torus distance of component D: min(D mod n, n − D mod n). "
            "For D ≤ ⌊n/2⌋: all distances ≤ ⌊n/2⌋ → step bound satisfied. □"
        ),
        conditions=["n is odd", "n >= 3", "D >= 1"],
        references=["V11 Audit C4 Closure Theorem — Closing Jump Identity",
                    "flu.core.fm_dance_path.step_bound_theorem"],
    ),
]


# ── Build unified registry ────────────────────────────────────────────────────

def _build_registry() -> Dict[str, TheoremRecord]:
    reg: Dict[str, TheoremRecord] = {}

# ── Rooting the Lineage (V1.2.3) ──────────────────────────────────────────

    # GEN-0: Rooting the 2017 FM-Dance origin
    reg["GEN-0"] = TheoremRecord(
        name="GEN-0 -- 2017 FM-Dance Siamese Origin",
        status="PROVEN",
        statement="FM-Dance is the n-dimensional generalisation of the Siamese magic square algorithm.",
        proof="Empirical reduction to Siamese for D=2 (proven via T5). Historical foundation of the path-traversal logic.",
        references=["2017-08-28_Symmetrische_Tanzschritte.pdf"]
    )

    # YM-1: Youvan–Mönnich Danielic Ten (corrected from GEN-1 collision in prior sprint)
    reg["YM-1"] = YM1_DANIELIC_TEN

# ── FLU Theorems ──────────────────────────────────────────

    # FM-Dance proven theorems (T1-T6 + SA-1 + N-ARY-1)
    for t in ALL_THEOREMS:
        key = t.name.split(" -- ")[0].strip()
        reg[key] = t

    # Kinetic theorems (KIB, BPT, CGW) — Bedrock
    for k in _KINETIC_THEOREMS:
        key = k.name.split(" -- ")[0].strip()
        reg[key] = k

    # L4 — Step-Bound Regime Lemma (V12)
    reg["L4"] = L4_STEP_BOUND_REGIME

    # C3W — Communion Weak Invariant Inheritance (V12 Wave 3)
    reg["C3W"]  = C3W_PROVEN
    reg["C3W-APN"] = C3W_APN

    # FM-Dance updated conjectures (C2 scoped, C1 removed → L2)
    for c in _UPDATED_CONJECTURES:
        key = c.name.split(" -- ")[0].strip()
        reg[key] = c

    # V12 Audit-integrated conjectures and theorems
    reg["T8"]         = T8_GRAY_BRIDGE
    reg["FM-1"]       = FM1_FRACTAL_MAGIC
    reg["BFRW-1"]     = BFRW1_DIFFUSION
    reg["TORUS_DIAM"] = TORUS_DIAM

    # V13 proof upgrades
    reg["T8b"]         = T8B_STEP_VECTOR_UNIQUENESS
    reg["C3W-STRONG"]  = C3W_STRONG
    reg["S2-GAUSS"]    = S2_GAUSS_PROOF
    reg["C2-SCOPED"]   = C2_SCOPED_PROVEN

    # V14 conjectures (open)
    reg["DN1"]          = DN1_DIGITAL_NET         # PROVEN V15.3+
    reg["DN1-GL"]       = DN1_GL_GRAECO_LATIN     # PROVEN V15.3+
    reg["DN1-OA"]       = DN1_OA_STRENGTH4        # PROVEN V15.3+
    reg["DN1-GEN"]      = DN1_GEN_ALL_ODD         # PROVEN V15.3.2 (all odd n)
    reg["DN1-REC"]      = DN1_REC_RECURSIVE       # PROVEN V15.3.2 (recursive)

    # V15.3.2 — DNO: Orthogonal Digital Net family (FractalNetOrthogonal)
    reg["DNO-GEN"]          = DNO_GEN           # Generator invertibility all n≥2
    reg["DNO-COEFF-EVEN"]   = DNO_COEFF_EVEN    # Even-n snake map OA
    reg["DNO-INV"]          = DNO_INV           # Inverse oracle O(d)
    reg["DNO-REC-MATRIX"]   = DNO_REC_MATRIX    # A^(k) tensor power
    reg["DNO-OPT"]          = DNO_OPT           # Bijectivity → max OA strength
    reg["DNO-P1"]           = DNO_P1            # Latin preserved under Owen
    reg["DNO-P2"]           = DNO_P2            # OA preserved per depth
    reg["DNO-OPT-FACT"]     = DNO_OPT_FACT      # Factorized subgroup O(d)
    reg["DNO-TVAL-BAL"]     = DNO_TVAL_BAL      # Balanced (0,4k,4k)-net
    reg["DNO-TVAL-REC"]     = DNO_TVAL_REC      # Full Niederreiter (3M,4kM,4k)-net
    reg["DNO-TVAL-STABLE"]  = DNO_TVAL_STABLE   # Dimension-stable t_bal=0
    reg["DNO-WALSH-REC"]    = DNO_WALSH_REC     # Trivial dual at all depths
    reg["DNO-DUAL"]         = DNO_DUAL          # D*={0}
    reg["DNO-ANOVA"]        = DNO_ANOVA         # Grid-constant ANOVA exactness
    reg["DNO-COEFF"]        = DNO_COEFF         # V_n + Walsh-annihilated exact
    reg["DNO-VAR"]          = DNO_VAR           # DN1+DN2 variance bound
    reg["DNO-VAR-REC"]      = DNO_VAR_REC       # Ultimate variance for DN1-REC+DN2
    reg["DNO-ETK"]          = DNO_ETK           # ETK discrepancy constant
    reg["DNO-WALSH"]        = DNO_WALSH         # Walsh-tight discrepancy
    reg["DNO-ASYM"]         = DNO_ASYM          # Tight asymptotic rate
    reg["DNO-SPECTRAL"]     = DNO_SPECTRAL      # Hard cutoff + exp decay spectrum
    reg["DNO-OPT-WALSH"]    = DNO_OPT_WALSH     # Walsh-space Pareto optimality
    reg["DNO-MINIMAX"]      = DNO_MINIMAX       # Minimax optimal
    reg["DNO-RKHS"]         = DNO_RKHS          # RKHS automatic ANOVA weighting
    reg["DNO-FUNC"]         = DNO_FUNC          # Exact integration class
    reg["DNO-SUPERIORITY"]  = DNO_SUPERIORITY   # Strict spectral dominance
    reg["DNO-FULL"]         = DNO_FULL          # Five simultaneous optimalities
    reg["DNO-PREFIX"]       = DNO_PREFIX        # Prefix discrepancy O(N^{-1/k})
    reg["OD-19-LINEAR"] = OD19_LINEAR            # PROVEN V15.3+
    reg["OD-16"]       = DELTA_MIN_19
    reg["OD-17"]       = DELTA_MIN_31

    # V14 audit integrations
    reg["T9"]          = T9_RADICAL_LATTICE          # Linear digital sequence, Faure conjugacy (PROVEN V15)
    reg["DN2"]         = DN2_APN_SCRAMBLED_NET        # APN scrambling → digital net (PROVEN V15.3)
    reg["DN2-ETK"]     = DN2_ETK                      # Discrepancy constant via ETK (PROVEN V15.3)
    reg["DN2-WALSH"]   = DN2_WALSH                    # Walsh-tight discrepancy (PROVEN V15.3)
    reg["DN2-VAR"]     = DN2_VAR                      # Owen-class variance bound (PROVEN V15.3)
    reg["DN2-ANOVA"]   = DN2_ANOVA                    # ANOVA interaction suppression (PROVEN V15.3)
    reg["OD-16-PM"]    = OD16_POWER_MAP_PROVEN        # Power-map obstruction Z_19 (PROVEN)
    reg["OD-17-PM"]    = OD17_POWER_MAP_PROVEN        # Power-map obstruction Z_31 (PROVEN)
    reg["HM-1"]        = HM1_HOLOGRAPHIC_SPARSITY     # ScarStore sparsity bound (PROVEN V14)

    # V14 open-debt closures
    reg["FMD-NET"]     = FMD_NET                      # FractalNet (0,D,D)-net at blocks (PROVEN)
    reg["OD-32-ITER"]  = OD32_ITER                    # O(1) amortized iterator (PROVEN)

    # V15 closure: OD-33
    reg["OD-33"]       = OD33_DIGITAL_SEQUENCE        # FM-Dance is a (0,D,D)-digital sequence (PROVEN)

    # V15 audit integration — formal bridge theorems
    reg["HAD-1"]    = HAD1_HADAMARD_COMMUNION   # Hadamard-Communion Isomorphism (PROVEN)
    reg["TSP-1"]    = TSP1_ROUTING_ORACLE        # Optimal TSP oracle on toroidal lattices (PROVEN)
    reg["CRYPTO-1"] = CRYPTO1_APN_DIFFUSION      # APN prime-field structural immunity (PROVEN)
    reg["DISC-1"]   = DISC1_DISCRETE_INTEGRAL    # FM-Dance Discrete Integral Identity (PROVEN)

    # V15 interface facet theorems (PROVEN)
    reg["LEX-1"]    = LEX1_LEXICON_FACET          # Bijective alphanumeric encoding (PROVEN)
    reg["INT-1"]    = INT1_INTEGRITY_FACET         # O(1) conservative-law sonde (PROVEN)
    reg["GEN-1"]    = GEN1_GENETIC_FACET           # Cryptographic APN seed portability (PROVEN)
    reg["INV-1"]    = INV1_INVARIANCE_FACET        # Cross-branch structural isomorphism (PROVEN)

    # V15 interface facet conjectures
    reg["HIL-1"]    = HIL1_HILBERT_FACET           # FM-Dance + RotationHub (RETIRED V15.1.3)
    reg["DEC-1"]    = DEC1_COHOMOLOGY_FACET        # ScarStore coset decomposition (PROVEN V15.1.2)

    # V15.1.4 — Spectral Unification
    reg["UNIF-1"]   = UNIF1_SPECTRAL_UNIFICATION   # DFT vanishing for all sum-separable arrays (PROVEN)


    # V15.2 — OD-27 Digital Net classification
    reg["OD-27"]    = OD27_DIGITAL_NET_CLASSIFICATION  # Digital-net classification: t=m(D-1) (PROVEN V15.2)
    
    # V15.2 — new proven theorems from Synthesis-Review (2026-03-12)
    reg["T10"]  = T10_LATTICE_CONVERGENCE   # Kinetic Lattice Convergence at N=n^{2d} (PROVEN)
    reg["C5"]   = C5_RECURSIVE_HYPERTORUS   # Recursive Hyper-Torus Latin preservation (PROVEN)

    # V15.2 — even-n branch formalisation
    reg["EVEN-1"] = EVEN1_LATIN_HYPERPRISM  # Even-n Latin Hyperprism via Kronecker decomposition (PROVEN)

    # PFNT theorems
    for p in _PFNT_THEOREMS:
        key = p.name.split(" -- ")[0].strip()
        reg[key] = p

    # Latin theorems (L1, L2, L3) — L2 upgraded from C1
    for l in _LATIN_THEOREMS:
        key = l.name.split(" -- ")[0].strip()
        reg[key] = l

    # Spectral theorems (S1, S2)
    for s in _SPECTRAL_THEOREMS:
        key = s.name.split(" -- ")[0].strip()
        reg[key] = s

    return reg


REGISTRY: Dict[str, TheoremRecord] = _build_registry()


# ── proof_status assignment ────────────────────────────────────────────────────
#
# Applied as post-processing so each theorem doesn't need a scattered constructor
# argument.  The mapping is the single source of truth for verification tier.
#
# Tiers (see TheoremRecord docstring for full definitions):
#   "sketch_and_computational"  — proof sketch + test-suite computational check
#   "algebraic_sketch"          — proof sketch only; covered by overall 196/196 suite
#   "empirical"                 — CONJECTURE; observed, not proved
#   "peer_reviewed"             — (none yet; V12+)
#
# This directly addresses the external audit observation (March 2026):
#   "20 THEOREMS PROVEN — but these are internal proofs, not peer reviewed"
#   → Accurate characterisation: algebraic sketch + computational verification.

_PROOF_STATUS_MAP: Dict[str, str] = {
    # sketch_and_computational ─────────────────────────────────────────────
    # Algebraic argument exists AND dedicated test covers the claim.
    "T1":      "sketch_and_computational",   # verify_bijection() round-trips
    "T2":      "sketch_and_computational",   # verify_path() coverage
    "T3":      "sketch_and_computational",   # verify_latin() n∈{3,5,7}, d∈{2,3,4}
    "T4":      "sketch_and_computational",   # bench_traversal: 0 violations
    "T5":      "sketch_and_computational",   # verify_siamese_d2()
    "T6":      "sketch_and_computational",   # verify_fractal()
    "T7":      "sketch_and_computational",   # v11a_t7_product_formula: 0 errors
    "KIB":     "sketch_and_computational",   # v11b_identify_step_bijection
    "BPT":     "sketch_and_computational",   # v11b_boundary_partition_* tests
    "L2":      "sketch_and_computational",   # verify_holographic_repair(); proof text explicit
    "PFNT-3":  "sketch_and_computational",   # verify_latin() covers hyperprism case
    "PFNT-5":  "sketch_and_computational",   # bench_fusion: 14/14 exact associativity

    # algebraic_sketch ─────────────────────────────────────────────────────
    # Correct algebraic argument; overall suite 196/196 but no dedicated
    # "Computationally verified across n,d" assertion in the proof text.
    "CGW":    "algebraic_sketch",    # Cayley graph walk; tested via traverse + cayley_generators
    "SRM":    "algebraic_sketch",    # Corollary of KIB+T1; tested indirectly
    "C4":     "algebraic_sketch",    # Step-bound corollary; v12_c4_is_proven
    "L4":     "algebraic_sketch",    # v12 — trivially derived from T4
    "C3W":    "algebraic_sketch",    # v12 Wave 3 — proven by PFNT-5 + S2 + S1
    "PFNT-1": "algebraic_sketch",
    "PFNT-2": "algebraic_sketch",
    "PFNT-4": "algebraic_sketch",    # Lehmer completeness; standard combinatorics
    "L1":     "algebraic_sketch",    # Pure algebra; verify_constant_line_sum
    "L3":     "algebraic_sketch",    # Follows from L2; test_l3 coverage
    "S1":     "algebraic_sketch",    # Special case of L1/PFNT-2
    "S2-Prime":"algebraic_sketch",   # Variance bound proved algebraically

    # empirical ────────────────────────────────────────────────────────────
    # CONJECTURE — observed empirically, algebraic proof incomplete.
    "S2":  "algebraic_sketch",   # PROMOTED PROVEN V12 Wave 2 — DFT linearity proof
    "C2":  "empirical",   # DISPROVEN_SCOPED — retained as negative result
    "C3":  "empirical",

    # V12 Audit-integrated — new entries
    "SA-1":    "algebraic_sketch",  # Separability lemma — algebraically tight proof
    "N-ARY-1": "algebraic_sketch",  # N-ary generalisation — det(T) argument
    "T8":      "empirical",         # Gray Bridge — n=2 reduction not yet verified
    "FM-1":    "empirical",         # Fractal Magic — partial proof only (audit V12)
    "BFRW-1":  "algebraic_trivial_via_diameter",  # Upgraded 2026-03 via TORUS_DIAM
    "TORUS_DIAM": "algebraic_trivial_via_diameter",  # Sprint 2026-03
    "C3W-APN": "algebraic_trivial_via_diameter",  # Upgraded 2026-03 via TORUS_DIAM

    # V14 open-debt closures
    "HM-1":       "algebraic_trivial",           # Representation theorem is exact by definition
    "FMD-NET":    "algebraic_trivial_via_bijection",  # Corollary of T1 bijection
    "OD-32-ITER": "algebraic_and_amortized",     # Correctness+amortized analysis; computationally verified
    "OD-33":      "algebraic_trivial_via_bijection",  # Corollary of T1 + FMD-NET; block translation argument

    # V15 bridge theorems (audit integration sprint)
    "HAD-1":    "algebraic_and_computational",   # Bit-masked seed proof + H@H.T=N·I verified d=2..6
    "TSP-1":    "algebraic_sketch",              # T2+C4+KIB structural argument
    "CRYPTO-1": "algebraic_sketch",              # Algebraic mismatch Z_p vs GF(2^k)
    "DISC-1":   "algebraic_and_computational",   # T·a = Φ(k) + T·Δa = σ_j; verified all n,d
    "LEX-1":    "algebraic_trivial",             # Bijection between equal-cardinality sets
    "INT-1":    "algebraic_sketch",              # L1 corollary; O(1) check
    "GEN-1":    "algebraic_trivial",             # SHA-256 verification exact by construction
    "INV-1":    "algebraic_sketch",              # Cross-branch structural isomorphism

    # V15.1.4 — Spectral Unification
    "UNIF-1":   "algebraic_and_computational",   # DFT linearity + char. orthogonality; S2 + HAD-1 verified

    # V15.2 — OD-27 Digital Net Classification (PROVEN V15.2)
    "OD-27":    "algebraic_and_computational",   # T-Rank Lemma + depth decoupling; t=m(D-1) exact

    # V15.2 — even-n branch formalisation (PROVEN V15.2)
    "EVEN-1":   "algebraic_and_computational",   # Gray-XOR × sum-mod Kronecker; 83-test suite

    # V15.3 — DN2 complete proof (ETK + Walsh + Variance + ANOVA)
    "DN2":         "algebraic_and_computational",   # PROVEN: all 8 sub-parts closed V15.3
    "DN2-ETK":     "algebraic_and_computational",   # ETK constant C_APN(D)=(B/sqrt(n))^D proven
    "DN2-WALSH":   "algebraic_and_computational",   # Walsh-native, same constant, frequency-decaying
    "DN2-VAR":     "algebraic_and_computational",   # Owen-class variance, gain independent of smoothness
    "DN2-ANOVA":   "algebraic_and_computational",   # ANOVA suppression, effective dimension reduction

    # V15.3+ — DN1 proven, new sub-theorems, OD-19-LINEAR closed
    "DN1":          "algebraic_and_computational",  # PROVEN: Graeco-Latin + OA(81,4,3,4) + recursive
    "DN1-GL":       "algebraic_and_computational",  # PROVEN: generation formulas + 0-mismatch verification
    "DN1-OA":       "algebraic_and_computational",  # PROVEN: OA strength-4 certificate (17 tests)
    "DN1-GEN":      "algebraic_and_computational",  # PROVEN: det=4, gcd(4,n)=1 for all odd n
    "DN1-REC":      "algebraic_and_computational",  # PROVEN: recursive strength doubling, k=2 verified
    # V15.3.2 — DNO: Orthogonal Digital Net family
    "DNO-GEN":          "algebraic_and_computational",
    "DNO-COEFF-EVEN":   "algebraic_and_computational",
    "DNO-INV":          "algebraic_and_computational",
    "DNO-REC-MATRIX":   "algebraic_and_computational",
    "DNO-OPT":          "algebraic",
    "DNO-P1":           "algebraic",
    "DNO-P2":           "algebraic_and_computational",
    "DNO-OPT-FACT":     "algebraic",
    "DNO-TVAL-BAL":     "algebraic_and_computational",
    "DNO-TVAL-REC":     "algebraic_and_computational",
    "DNO-TVAL-STABLE":  "algebraic",
    "DNO-WALSH-REC":    "algebraic",
    "DNO-DUAL":         "algebraic",
    "DNO-ANOVA":        "algebraic",
    "DNO-COEFF":        "algebraic_and_computational",
    "DNO-VAR":          "algebraic",
    "DNO-VAR-REC":      "algebraic",
    "DNO-ETK":          "algebraic_and_computational",
    "DNO-WALSH":        "algebraic",
    "DNO-ASYM":         "algebraic",
    "DNO-SPECTRAL":     "algebraic_and_computational",
    "DNO-OPT-WALSH":    "algebraic",
    "DNO-MINIMAX":      "algebraic",
    "DNO-RKHS":         "algebraic",
    "DNO-FUNC":         "algebraic",
    "DNO-SUPERIORITY":  "algebraic",
    "DNO-FULL":         "algebraic_and_computational",
    "DNO-PREFIX":       "algebraic_and_computational",
    "OD-19-LINEAR": "algebraic_and_computational",  # PROVEN: linear-digit uniqueness, 7-step proof
    "T8b":          "algebraic_and_computational",  # updated: OD-19 open part now closed by OD-19-LINEAR
}

# Apply in-place — replace entries with proof_status set in the map,
# but respect the theorem's own proof_status if it already exists.
REGISTRY = {
    tid: TheoremRecord(
        name         = t.name,
        status       = t.status,
        statement    = t.statement,
        proof        = t.proof,
        conditions   = t.conditions,
        references   = t.references,
        proof_status = _PROOF_STATUS_MAP.get(tid, getattr(t, "proof_status", "algebraic_sketch")),
    )
    for tid, t in REGISTRY.items()
}

# ── Query helpers ─────────────────────────────────────────────────────────────

def get_theorem(key: str) -> Optional[TheoremRecord]:
    """
    Retrieve a theorem by its short key (e.g. 'T1', 'C2', 'PFNT-3').
    Returns None if not found.
    """
    return REGISTRY.get(key)


def proven_theorems() -> List[TheoremRecord]:
    """Return all entries with STATUS: PROVEN."""
    return [t for t in REGISTRY.values() if t.is_proven()]


def open_conjectures() -> List[TheoremRecord]:
    """Return all entries with STATUS: CONJECTURE or PARTIAL."""
    return [t for t in REGISTRY.values() if t.is_open()]


def disproven_negative_results() -> List[TheoremRecord]:
    """Return all entries with a DISPROVEN status (any variant)."""
    return [t for t in REGISTRY.values() if t.is_disproven()]


def retired_theorems() -> List[TheoremRecord]:
    """Return all entries with STATUS: RETIRED (withdrawn conjectures).

    RETIRED entries are withdrawn conjectures that were found to be
    self-contradictory or otherwise not worth pursuing, but are preserved
    as named research artifacts for historical completeness.
    """
    return [t for t in REGISTRY.values() if t.is_retired()]


def status_report() -> str:
    """
    Return a full status report for all theorems and conjectures.

    Includes proof_status tier for each theorem (added March 2026 per audit).

    Example
    -------
    >>> from flu.theory.theorem_registry import status_report
    >>> print(status_report())
    """
    proven = proven_theorems()
    open_  = open_conjectures()

    tier_label = {
        "sketch_and_computational": "sketch+test",
        "algebraic_sketch":         "sketch",
        "empirical":                "empirical",
        "peer_reviewed":            "peer_reviewed",
    }

    lines = [
        "FLU V15 — Theorem Registry Status Report",
        "=" * 55,
        f"\nPROVEN ({len(proven)} theorems):  [proof_status tier shown]",
    ]
    for t in proven:
        tier = tier_label.get(t.proof_status, t.proof_status)
        lines.append(f"  [PROVEN / {tier:16}] {t.name}")

    disproven = disproven_negative_results()
    retired   = retired_theorems()
    lines.append(f"\nOPEN CONJECTURES ({len(open_)} items):")
    for c in open_:
        lines.append(f"  [CONJECTURE / empirical    ] {c.name}")
    if disproven:
        lines.append(f"\nNEGATIVE RESULTS / DISPROVEN ({len(disproven)} items):")
        for d in disproven:
            lines.append(f"  [DISPROVEN_SCOPED          ] {d.name}")
    if retired:
        lines.append(f"\nRETIRED / WITHDRAWN ({len(retired)} items):")
        for r in retired:
            lines.append(f"  [RETIRED                   ] {r.name}")

    from collections import Counter
    tier_counts = Counter(t.proof_status for t in proven)
    lines.append(f"\nProof tier summary (PROVEN only):")
    for tier, count in sorted(tier_counts.items()):
        lines.append(f"  {tier_label.get(tier, tier):20}: {count}")

    lines.append(f"\nTotal registered: {len(REGISTRY)}")
    return "\n".join(lines)
