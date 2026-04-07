from flu._version import __version__, FLU_VERSION, FLU_VERSION_LABEL  # noqa: F401
__author__  = "Felix Mönnich & The Kinship Mesh Collective"

"""FLU — Universal Fractal Logic Unit

A mathematically rigorous library for n-ary Latin hyperprisms, toroidal traversals,
quasi-Monte Carlo sequences, orthogonal designs, sparse compression, and related
combinatorial structures.

FLU provides a unified algebraic framework with direct O(D) bijections, Latin
preservation, Hamiltonian paths on tori, and interfaces backed by 58 proven
theorems (as of V15, March 2026).

Core capabilities include:
- FM-Dance bijection and path traversal (O(D) coordinate access)
- Latin hypercube generation (odd/even n)
- Hadamard matrix generation via Communion isomorphism
- n-ary Gray codes and tuned space-filling curves
- Discrete exterior calculus operations on toroidal manifolds
- Sparse infinite-scale manifolds with lossless holographic repair
- Sparse Arithmetic Stack (OPER-1) for lazy field calculus

All high-level features are exposed through clean, theorem-linked interfaces.

Quick-start example
-------------------
import flu

# 1. Get any point on an infinite-scale sparse manifold (O(D) memory/time)
M = flu.manifold(n=3, d=64, sparse=True)           # 3^64 cells, ~1.5 kB RAM
print(M[0, 1, -1, 2, ..., -1])                      # arbitrary coordinate lookup

# 2. Traverse with O(1) amortized iterator
it = flu.traversal.FMDanceIterator(n=5, d=4)
for _ in range(10):
    print(next(it))                                 # next point in path

# 3. Generate Hadamard matrix with proof provenance
H = flu.HadamardFacet(d=6).generate()               # 64×64 orthogonal matrix
print(H @ H.T)                                      # should be 64·I

# 4. Check theorem status
print(flu.theory.status_report())                   # "58 PROVEN, 5 CONJECTURE, ..."
t = flu.theory.get_theorem("HAD-1")
print(t.info())                                     # theorem name & status """

from flu._version import __version__, FLU_VERSION, FLU_VERSION_LABEL  # noqa: F401
__author__  = "Felix Mönnich & The Kinship Mesh Collective"

from flu.theory                import PhasedFractalNumberTheory
from flu.theory.theorem_registry import (
    REGISTRY, get_theorem, proven_theorems, open_conjectures,
    disproven_negative_results, status_report,
)
from flu.theory.theory_fm_dance    import (
    verify_l4_step_bound_regimes, verify_step_bound_under_communion,
    C3W_PROVEN, C3W_APN,
    DISC1_DISCRETE_INTEGRAL, verify_discrete_integral_identity,
)
from flu.theory.theory_latin   import (
    holographic_repair, verify_holographic_repair,
    verify_constant_line_sum, verify_all_latin_theorems,
    line_sum_constant, byzantine_fault_tolerance_degree,
)
from flu.theory.theory_spectral import (
    verify_spectral_flatness, verify_dc_zero, compute_spectral_profile,
)
from flu.theory.theory_container import (
    PermutationLattice, GENERATOR_ROLES,
    PERMUTATION_LATTICE_ALGEBRA, permutation_lattice_summary,
)
from flu.core.operators import (
    FLUOperator, TMatrixOperator, APNPermuteOperator, RotationHubOperator, ExternalPhysics,
)
from flu.core.fm_dance         import index_to_coords, coords_to_index, generate_fast, verify_bijection
from flu.core.fm_dance_path    import (
    path_coord, path_coord_to_rank, traverse, traverse_reverse,
    generate_path_array, verify_all as verify_path,
    step_bound_theorem, verify_siamese_d2, verify_fractal,
    identify_step, step_vector, invert_fm_dance_step,
    inverse_step_vector, cayley_generators, cayley_inverse_generators,
    boundary_partition_sizes, fractal_fault_lines,
    FMDanceIterator,          # OD-32: O(1) amortized incremental traversal
)
from flu.core.factoradic       import (
    random_apn_search, factoradic_unrank, factoradic_rank, arrow_generator,
    factoradic_to_fm_coords, fm_coords_to_factoradic, ArrowStep,
    differential_uniformity, nonlinearity_score, is_pn_permutation,
    unrank_optimal_seed, GOLDEN_SEEDS,
)
from flu.core.lo_shu           import LoShuHyperCell, Perspective, CellStrata
from flu.core.lo_shu_sudoku    import LoShuSudokuHyperCell, make_hypercell as make_sudoku_hypercell
from flu.core.even_n           import generate as even_n_generate, verify as even_n_verify
from flu.core.hypercell        import FLUHyperCell
from flu.core.fractal_3_6      import FractalHyperCell_3_6, CellPair, MicroCell, SudokuMacroAdapter
from flu.core.parity_switcher  import generate, generate_metadata, verify_latin
from flu.container.contract    import UKMCContract
from flu.container.manifold    import cell_to_sparse_coords, sparse_coords_to_norm0, verify_seam
from flu.container.communion   import CommunionEngine
from flu.container.sparse      import SparseCommunionManifold, SparseArithmeticManifold, SparseEvenManifold, SparseOrthogonalManifold, ScarStore, ForeignField
from flu.applications.codes      import LatinSquareCode, build_code_matrix
from flu.applications.design     import ExperimentalDesign, DesignResult
from flu.applications.neural     import FLUInitializer, DynamicFLUNetwork
from flu.applications.quantum    import TensorNetworkSimulator
from flu.applications.lighthouse import LighthouseBeacon, BeaconKey, cli_main
from flu.core.vhdl_gen         import generate_vhdl, export_vhdl
from flu.core.fractal_net      import FractalNet, FractalNetKinetic, FractalNetOrthogonal   # OD-27 / T9 / DN1: identity / T-matrix / OA(n⁴,4,n,4) digital nets
from flu.interfaces.digital_net import (
    FractalNetCorputFacet,    # FMD-NET PROVEN — van der Corput facet (researcher interface)
    FractalNetKineticFacet,   # T9 PROVEN      — kinetic digital net facet (researcher interface)
    FractalNetOrthogonalFacet,# DNO-REC PROVEN — orthogonal digital net facet (researcher interface)
)
from flu.theory.theory_spectral import (
    spectral_dispersion_bound, SpectralDispersionBound,
)
from flu.utils.benchmarks      import full_benchmark_report, spectral_probe_large_n

# ── N-ary generalisation (V12 Sprint) ─────────────────────────────────────────
from flu.core.n_ary import (
    nary_info, nary_generate, nary_generate_signed, nary_verify,
    nary_step_bound, nary_comparison_table, recommend_base,
    ternary_block_base, verify_nary_bijection,
)
from flu.theory.theory_communion_algebra import (
    run_communion_algebra_investigation,
    classify_structure,
    phi_add, phi_max, phi_lex_ordered,
)

from flu.interfaces import (
    GrayCodeFacet,
    HadamardFacet,
    CurveFacet,
    CohomologyFacet,
    DesignFacet,
    NeuralFacet,
    CryptoFacet,
    IntegrityFacet,
    GeneticFacet,
    InvarianceFacet,
    LexiconFacet,
    # QMCFacet when ready
)

# ── Sub-namespace aliases (OD-10 / Sprint Item 10) ────────────────────────────
# All existing flat exports remain; these are ADDITIONAL convenience namespaces.

class _Namespace:
    """Lightweight namespace object for sub-module access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        keys = sorted(self.__dict__.keys())
        return f"flu.{getattr(self, '_name', '?')}({', '.join(keys)})"


traversal = _Namespace(
    traverse          = traverse,
    traverse_reverse  = traverse_reverse,
    path_coord        = path_coord,
    path_coord_to_rank= path_coord_to_rank,
    FMDanceIterator   = FMDanceIterator,  # OD-32: O(1) amortized iterator
)
traversal._name = "traversal"

latin = _Namespace(
    generate          = generate,
    generate_fast     = generate_fast,
    verify_latin      = verify_latin,
    holographic_repair= holographic_repair,
)
latin._name = "latin"

seeds = _Namespace(
    GOLDEN_SEEDS       = GOLDEN_SEEDS,
    unrank_optimal_seed= unrank_optimal_seed,
    factoradic_unrank  = factoradic_unrank,
    factoradic_rank    = factoradic_rank,
    random_apn_search  = random_apn_search,
)
seeds._name = "seeds"

theory = _Namespace(
    get_theorem      = get_theorem,
    proven_theorems  = proven_theorems,
    open_conjectures = open_conjectures,
    status_report    = status_report,
)
theory._name = "theory"

nary = _Namespace(
    info             = nary_info,
    generate         = nary_generate,
    generate_signed  = nary_generate_signed,
    verify           = nary_verify,
    step_bound       = nary_step_bound,
    comparison_table = nary_comparison_table,
    recommend_base   = recommend_base,
    ternary_block    = ternary_block_base,
)
nary._name = "nary"

operators = _Namespace(
    Base=FLUOperator, TMatrix=TMatrixOperator,
    APNPermute=APNPermuteOperator, External=ExternalPhysics, _name="operators"
)

# ── Unified Exports ───────────────────────────────────────────────────────────

__all__ = [
    # Factory & Orchestration
    "manifold", "generate", "FLUHyperCell", "CommunionEngine",
    # Traversal & Addressing
    "traverse", "path_coord", "FMDanceIterator",
    # QMC & Nets
    "FractalNet", "FractalNetKinetic",
    # Sparse & Arithmetic
    "ScarStore", "SparseCommunionManifold", "SparseArithmeticManifold", "SparseEvenManifold", "SparseOrthogonalManifold", "ForeignField",
    # Logic & Theory
    "get_theorem", "status_report", "FLUOperator",
    # Namespaces
    "traversal", "latin", "seeds", "theory", "nary", "operators",
    # Major Facets / Apps
    "ExperimentalDesign", "FLUInitializer", "LighthouseBeacon", 
    "HadamardFacet", "LexiconFacet", "CryptoFacet", "GrayCodeFacet"
]

# ── Unified manifold factory ──────────────────────────────────────────────────
 
def manifold(n: int, d: int, sparse: bool = False, signed: bool = True):
    """
    Unified manifold factory — returns either a dense array or a sparse oracle.
 
    Dispatch table
    --------------
    sparse=False  →  dense ndarray via parity_switcher.generate()
                     Works for any n ≥ 2.  Shape (n,)*d.
 
    sparse=True, n odd   →  SparseCommunionManifold  (O(D·n) memory)
                             Seeds chosen from APN/PN-class Golden Seed table.
                             Requires odd n for exact mean-centring (PFNT-2).
 
    sparse=True, n even  →  SparseEvenManifold  (O(D) memory, parameter-free)
                             Uses Gray-coded XOR micro-block + sum-mod macro-block
                             (Kronecker product, EVEN-1).  No seed lookup needed.
 
    Parameters
    ----------
    n      : int   base radix (≥ 2, odd or even)
    d      : int   spatial dimension (≥ 1)
    sparse : bool  True → lazy O(D) oracle; False (default) → dense ndarray
    signed : bool  True (default) → values in [−⌊n/2⌋, ⌊n/2⌋−1] (or ⌊n/2⌋ for odd n)
                   False → values in [0, n−1]
                   For sparse manifolds this controls the *output* range;
                   input coordinates are always signed.
 
    Returns
    -------
    SparseCommunionManifold  if sparse=True and n is odd
    SparseEvenManifold       if sparse=True and n is even
    np.ndarray               if sparse=False  (shape (n,)*d)
 
    Examples
    --------
    Dense (materialised):
        M = flu.manifold(5, 3)               # odd n,  shape (5,5,5)
        M = flu.manifold(6, 2)               # even n, shape (6,6)
 
    Sparse, odd n (holographic oracle):
        M = flu.manifold(3, 64, sparse=True) # 3^64 cells, ~1.5 kB RAM
        val = M[0, 1, -1, ...]               # O(D) lookup, signed coords
 
    Sparse, even n (parameter-free oracle):
        M = flu.manifold(4, 32, sparse=True) # 4^32 cells, O(D) memory
        val = M[(-1, 0, 1, ...)]             # O(D) lookup, signed coords
 
    Unsigned output:
        M = flu.manifold(6, 2, sparse=True, signed=False)
        val = M[(-3, 2)]                     # signed coords, returns 0..5
    """
    from flu.utils.math_helpers import is_odd
 
    if sparse:
        if is_odd(n):
            from flu.core.factoradic import unrank_optimal_seed
            # Unique APN/PN-class seeds per axis to suppress inter-axis correlation
            seeds_list = [unrank_optimal_seed(i % 8, n, signed=False) for i in range(d)]
            return SparseCommunionManifold(n=n, seeds=seeds_list)
        else:
            # Even n: parameter-free Kronecker oracle — no seed lookup required
            return SparseEvenManifold(n=n, d=d, signed=signed)
 
    return generate(n=n, d=d, signed=signed)
 
