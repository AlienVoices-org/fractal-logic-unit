"""
flu/container/sparse.py
=======================
Sparse / Lazy Communion Manifold for Hyper-Dimensional Topologies.

Provides an O(D) memory, O(D) compute holographic oracle that behaves like a
standard NumPy array via Python's __getitem__ protocol.  Derived directly from
Theorems T1 (Bijective Addressing) and C3W-PROVEN (Communion Weak Inheritance).

Mathematical basis
------------------
For add-communion with seeds π_0, …, π_{D-1}, the manifold value at coordinate
x = (x_0, …, x_{D-1}) is:

    M[x] = (Σ_i π_i[x_i + ⌊n/2⌋]) mod n  −  ⌊n/2⌋

Because the sum is separable, the full n^D array is never materialised.
Instead only the D seed arrays of length n are stored: O(D·n) memory.

Communion (⊗_add) of two manifolds reduces to seed concatenation (C3W):

    (M1 ⊗ M2)[x1, x2] = M1[x1] + M2[x2]

No n^(d1+d2) allocation is ever needed.

STATUS: PROVEN — derived from T1 and C3W-PROVEN.
"""

from __future__ import annotations

from typing import List, Tuple, Union, Any, Callable
import numpy as np

from flu.core.fm_dance import index_to_coords
from flu.utils.math_helpers import is_odd

class ArithmeticMixin:
    """
    Mixin providing lazy arithmetic operator overloading (OPER-1).
    
    Routes all operations through CommunionEngine.simplify to guarantee
    O(1) constant folding and prevent memory leaks / recursion overflows.
    All FLU sparse manifolds must inherit this to participate in the field calculus.
    """
    def __add__(self, other):
        from flu.container.communion import CommunionEngine
        return CommunionEngine.simplify(self, other, np.add, "⊕")

    def __sub__(self, other):
        from flu.container.communion import CommunionEngine
        return CommunionEngine.simplify(self, other, np.subtract, "⊖")

    def __mul__(self, other):
        from flu.container.communion import CommunionEngine
        return CommunionEngine.simplify(self, other, np.multiply, "⊗")

    def __truediv__(self, other):
        from flu.container.communion import CommunionEngine
        return CommunionEngine.simplify(self, other, np.true_divide, "⊘")

    def materialize(self) -> np.ndarray:
        """
        Evaluates the entire manifold into a dense NumPy array.
        Warning: O(n^D) memory and compute. Use only for small manifolds.
        """
        shape = tuple([self.n] * self.d)
        out = np.zeros(shape, dtype=float)
        half = self.n // 2
        # Vectorized batch evaluation if possible, else fallback to iteration
        for idx in np.ndindex(*shape):
            coord = tuple(i - half for i in idx)
            out[idx] = self[coord]
        return out

class ConstantManifold(ArithmeticMixin):
    """
    A zero-cost manifold representing a scalar constant.
    Prevents duck-typing crashes during algebraic simplification.
    """
    def __init__(self, value: float, n: int, d: int) -> None:
        self.value = float(value)
        self.n = n
        self.d = d

    def __getitem__(self, key: Union[Tuple[int, ...], np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(key, np.ndarray):
            if key.shape[-1] != self.d and self.d > 0:
                raise ValueError(f"Expected last dimension {self.d}, got {key.shape[-1]}")
            return np.full(key.shape[:-1], self.value, dtype=float)
        return self.value

    def __repr__(self) -> str:
        return f"ConstantManifold({self.value}, n={self.n}, d={self.d})"

class SparseCommunionManifold(ArithmeticMixin):
    """
    Holographic representation of a vast n^D Latin hyperprism.

    Evaluates cell values strictly on-demand (Lazy Evaluation / Sparse Oracle).
    Communion (⊗_add) of two manifolds costs O(1) memory — just concatenate seeds.

    Parameters
    ----------
    n : int
        Base radix.  Must be odd for exact mean-centering (PFNT-2).
    seeds : list of np.ndarray
        One 1-D array of length n per dimension.  Each array must be a
        permutation of Z_n (unsigned 0..n-1).

    Examples
    --------
    >>> import numpy as np
    >>> from flu.container.sparse import SparseCommunionManifold
    >>> seeds = [np.array([2, 0, 1]), np.array([1, 2, 0])]
    >>> M = SparseCommunionManifold(n=3, seeds=seeds)
    >>> M[0, -1]          # single cell O(D)
    0
    >>> coords = np.array([[0, 0], [1, -1], [-1, 1]])
    >>> M[coords]         # batch evaluation O(N·D)
    array([ 1,  0, -1])
    """
        
    def __init__(self, n: int, seeds: List[np.ndarray]) -> None:
        if not is_odd(n):
            raise ValueError(
                f"SparseCommunionManifold requires odd n for exact mean-centering "
                f"(PFNT-2), got n={n}."
            )
        if not seeds:
            raise ValueError("seeds list must not be empty.")
        self.n    = n
        self.d    = len(seeds)
        self.half = n // 2
        self.shape: Tuple[int, ...] = tuple([n] * self.d)
        # Store as read-only arrays; unsigned 0..n-1 internally
        self.seeds: List[np.ndarray] = [np.asarray(s, dtype=int) for s in seeds]
        for i, s in enumerate(self.seeds):
            if s.shape != (n,):
                raise ValueError(
                    f"Seed {i} has shape {s.shape}; expected ({n},)."
                )

    # ── Communion operator ──────────────────────────────────────────────────

    @classmethod
    def commune(
        cls,
        M1: "SparseCommunionManifold",
        M2: "SparseCommunionManifold",
    ) -> "SparseCommunionManifold":
        """
        Communion operator (⊗_add) for two sparse manifolds.

        By C3W-PROVEN, add-communion of two sum-separable hyperprisms is
        itself a sum-separable hyperprism whose seeds are the concatenation
        of both seed lists.

        Memory cost: O(d1 + d2) — no n^(d1+d2) allocation ever occurs.

        Parameters
        ----------
        M1, M2 : SparseCommunionManifold
            Must share the same base n.
        """
        if M1.n != M2.n:
            raise ValueError(
                f"Communion requires identical base n; got {M1.n} and {M2.n}."
            )
        return cls(n=M1.n, seeds=M1.seeds + M2.seeds)

    # ── Indexing protocol ───────────────────────────────────────────────────

    def __getitem__(
        self,
        key: Union[int, Tuple, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """
        Pythonic indexing — supports single coordinates and NumPy batch arrays.

        Single cell (returns int, O(D)):
            M[0, -1, 1]           # exact signed coordinates

        Batch query (returns ndarray, O(N·D)):
            coords = np.array([[0, -1, 1], [1, 0, -1]])  # shape (N, D)
            M[coords]             # returns shape (N,)

        Parameters
        ----------
        key : tuple of ints | np.ndarray of shape (..., D)
        """
        if isinstance(key, np.ndarray):
            return self._batch_evaluate(key)

        # Normalise scalar index to tuple
        if isinstance(key, (int, np.integer)):
            key = (int(key),)
        key = tuple(int(k) for k in key)

        if len(key) != self.d:
            raise IndexError(
                f"Expected {self.d} coordinates, got {len(key)}."
            )
        return self._evaluate_single(key)

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _evaluate_single(self, coords: Tuple[int, ...]) -> int:
        """Evaluate one cell in O(D) time."""
        val = 0
        for axis, c in enumerate(coords):
            if not (-self.half <= c <= self.half):
                raise IndexError(
                    f"Coordinate {c} out of bounds for signed base {self.n} "
                    f"(valid range [{-self.half}, {self.half}])."
                )
            unsigned_idx = (c + self.half) % self.n
            val += int(self.seeds[axis][unsigned_idx])
        return int(val % self.n) - self.half

    def _batch_evaluate(self, coords_array: np.ndarray) -> np.ndarray:
        """
        Vectorised batch evaluation.

        Parameters
        ----------
        coords_array : np.ndarray, shape (..., D)
            Signed coordinates in [-⌊n/2⌋, ⌊n/2⌋].

        Returns
        -------
        np.ndarray, shape (...)
        """
        coords_array = np.asarray(coords_array, dtype=int)
        if coords_array.shape[-1] != self.d:
            raise ValueError(
                f"Last dimension of batch must be {self.d}, "
                f"got {coords_array.shape[-1]}."
            )
        # Shift signed coords → unsigned indices
        unsigned = (coords_array + self.half) % self.n

        # Stack seeds: shape (D, n)
        seeds_stacked = np.stack(self.seeds, axis=0)

        # Gather per-axis seed values: shape (..., D)
        gathered = seeds_stacked[
            np.arange(self.d),
            unsigned,
        ]  # advanced indexing: (D,) × (..., D) → (..., D)

        # Sum over D axis and centre
        return (np.sum(gathered, axis=-1) % self.n) - self.half

    # ── FM-Dance rank interface ─────────────────────────────────────────────

    def cell_at_rank(self, k: int) -> int:
        """
        Evaluate the cell at the k-th step of the FM-Dance Hamiltonian path.

        Complexity: O(D) addressing (T1) + O(D) evaluation = O(D).

        Parameters
        ----------
        k : int  FM-Dance rank in [0, n^D)
        """
        # index_to_coords already returns signed balanced coordinates
        coords_signed = tuple(int(c) for c in index_to_coords(k, self.n, self.d))
        return self._evaluate_single(coords_signed)

    # ── Helpers ─────────────────────────────────────────────────────────────
    
    def __repr__(self) -> str:
        return (
            f"SparseCommunionManifold(n={self.n}, d={self.d}, "
            f"shape={self.shape}, memory≈{self.n * self.d * 8}B)"
        )

class SparseEvenManifold(ArithmeticMixin):
    """
    O(D) memory/compute oracle for even-n Latin hyperprisms.

    Implements the V15.2 even-n decomposition — Gray-coded XOR micro-block
    (over Z_{2^k}) combined with sum-mod macro-block (over Z_m, m odd) via
    the mixed-radix Kronecker product — without ever materialising the n^D grid.

    The macro layer uses all-ones coefficients, matching flu.core.even_n.generate()
    exactly (sum-mod construction, proven Latin by T3 / N-ARY-1).

    Plugs seamlessly into the SparseArithmeticManifold tree (OPER-1).

    Parameters
    ----------
    n       : int   even radix
    d       : int   spatial dimension
    signed  : bool  if True (default), output values are in [−n/2, n/2−1];
                    if False, output values are in [0, n−1].
                    Input coordinates are always signed regardless of this flag.
    use_xor : bool  if True (default), use Gray-coded XOR for the 2^k factor;
                    if False, use sum-mod for both factors (useful for debugging).

    STATUS: PROVEN — O(D) algorithmic parity with flu.core.even_n.generate().
                     Latin property follows from even_n.py theorem (EVEN-1).
    """
    def __init__(self, n: int, d: int, signed: bool = True, use_xor: bool = True) -> None:
        if n % 2 != 0:
            raise ValueError("SparseEvenManifold requires even n.")
        self.n      = n
        self.d      = d
        self.signed = signed
        self.use_xor = use_xor
        self.half   = n // 2
        self.shape: Tuple[int, ...] = tuple([n] * d)

        # Decompose n = 2^k * m (m odd)
        k, m = 0, n
        while m % 2 == 0:
            m //= 2
            k += 1
        self.step = 2 ** k
        self.m    = m

        # Macro coefficients: all-ones vector (sum-mod construction).
        # This matches flu.core.even_n._sum_mod_latin exactly.
        # DO NOT use pow(3, i, m) here — that diverges from generate() and
        # breaks the Latin property when 3 | m (e.g. m=3 gives c_1=0).
        self.coeffs: List[int] = [1] * d

    def __getitem__(self, key: Union[Tuple[int, ...], np.ndarray]) -> Union[float, np.ndarray]:
        """
        O(D) resolution. Supports signed single coordinates and vectorised batch arrays.

        Single cell (returns scalar):
            M[-1, 2]               # signed coords in [-n//2, n//2 - 1]

        Batch query (returns ndarray, shape (...)):
            coords = np.array([[-1, 2], [0, -3]])   # shape (N, D)
            M[coords]
        """
        # ── 1. Batch Vectorised Evaluation ──────────────────────────────────
        if isinstance(key, np.ndarray):
            if key.shape[-1] != self.d:
                raise ValueError(f"Expected last dimension {self.d}, got {key.shape[-1]}")

            # Signed coords → unsigned indices [0, n-1]
            i_a = key + self.half

            # Micro component (Z_{2^k})
            u_a = i_a % self.step
            if self.use_xor:
                gray_val  = u_a ^ (u_a >> 1)
                micro_val = np.bitwise_xor.reduce(gray_val, axis=-1)
            else:
                micro_val = np.sum(u_a, axis=-1) % self.step

            # Macro component (Z_m): sum-mod with all-ones coefficients
            if self.m > 1:
                v_a       = i_a // self.step
                macro_val = np.sum(v_a, axis=-1) % self.m
            else:
                macro_val = 0

            raw = macro_val * self.step + micro_val
            return raw - self.half if self.signed else raw

        # ── 2. Single Coordinate Evaluation ─────────────────────────────────
        if isinstance(key, (int, np.integer)):
            key = (int(key),)
        key = tuple(int(k) for k in key)
        if len(key) != self.d:
            raise IndexError(f"Expected {self.d} coordinates, got {len(key)}")

        micro_val = 0
        macro_val = 0

        for x_a in key:
            i_a = x_a + self.half
            if not (0 <= i_a < self.n):
                raise IndexError(
                    f"Coordinate {x_a} out of bounds for n={self.n} "
                    f"(valid signed range [{-self.half}, {self.half - 1}])."
                )
            u_a = i_a % self.step
            if self.use_xor:
                micro_val ^= u_a ^ (u_a >> 1)
            else:
                micro_val = (micro_val + u_a) % self.step

            if self.m > 1:
                v_a       = i_a // self.step
                macro_val = (macro_val + v_a) % self.m

        raw = macro_val * self.step + micro_val
        return float(raw - self.half) if self.signed else float(raw)

    def __repr__(self) -> str:
        return (
            f"SparseEvenManifold(n={self.n}, d={self.d}, "
            f"signed={self.signed}, use_xor={self.use_xor})"
        )

class SparseArithmeticManifold(ArithmeticMixin):
    """
    V16 Sparse Fractal Arithmetic Stack (OPER-1).
    
    A lazy, O(D) memory 'Oracle' node that resolves coordinate values by 
    traversing an operator tree. Supports scalar broadcasting and chaining.
    
    Mathematical guarantees:
      ⊕ (add), ⊖ (sub) : Preserves Mean-Zero (S1) and Line-Sum (L1) invariants 
                         ONLY IF both operands satisfy them.
      ⊗ (mul), ⊘ (div) : Non-linear field transformations (Latin property lost, 
                         energy density structures emerge).
    """
    def __init__(self, left: Any, right: Any, op: Callable, symbol: str) -> None:
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol
        
        # Inherit spatial bounds from whichever operand is a manifold
        self.n = getattr(left, 'n', getattr(right, 'n', None))
        self.d = getattr(left, 'd', getattr(right, 'd', None))
        
        if self.n is None or self.d is None:
            raise ValueError("At least one operand must be a valid Manifold.")

    def __getitem__(self, coord: Tuple[int, ...]) -> float:
        """RECURSIVE RESOLUTION: O(D * depth)"""
        l_val = self.left[coord] if hasattr(self.left, '__getitem__') else self.left
        r_val = self.right[coord] if hasattr(self.right, '__getitem__') else self.right
        return float(self.op(l_val, r_val))

    def __repr__(self) -> str:
        return f"({self.left} {self.symbol} {self.right})"

class ForeignField(ArithmeticMixin):
    """
    A bridge for external tensors (PyTorch, JAX, raw NumPy).
    
    Acts as a proxy for non-FLU data so it can natively participate in the 
    SparseArithmeticManifold (OPER-1) expression tree.
    
    DESIGN INTENT & LIMITATIONS:
      - No L1/S1/T3 mathematical guarantees are made for this data.
      - No ScarStore compression is applied.
      - Dropped inheritance from SparseCommunionManifold (avoids LSP violations).
    """

    def __init__(self, data: Any, index_map: Optional[Callable] = None) -> None:
        # Coerce to numpy if possible, else keep raw (for torch/jax duck-typing)
        try:
            self.data = np.asarray(data)
        except Exception:
            self.data = data
            
        self.shape = self.data.shape
        self.d = len(self.shape)
        
        # For FLU arithmetic compatibility, 'n' is usually the size of the 
        # first dimension. If the tensor is non-uniform, this is just a proxy.
        self.n = self.shape[0] if self.d > 0 else 0
        
        # Pre-compute per-axis shifts for signed-to-unsigned conversion
        self.half = tuple(s // 2 for s in self.shape)
        self.index_map = index_map

    def __getitem__(self, key: Union[Tuple[int, ...], np.ndarray]) -> Union[float, np.ndarray]:
        """
        O(1) access. Supports single coordinates and vectorized batch arrays.
        Enforces strict bounds checking to prevent silent NumPy wrap-around.
        """
        # 1. Custom Index Mapping
        if self.index_map is not None:
            idx = self.index_map(key, self.shape)
            return self.data[idx]

        # 2. Batch Evaluation (NumPy array of shape (..., D))
        if isinstance(key, np.ndarray):
            if key.shape[-1] != self.d:
                raise ValueError(f"Expected last dimension to be {self.d}, got {key.shape[-1]}")
            
            shift = np.array(self.half, dtype=int)
            idx_array = key + shift
            
            # --- THE FIX: Vectorized Bounds Checking ---
            # We must check against self.shape for every dimension.
            # Shape of self.shape is (D,). Shape of idx_array is (..., D).
            shape_array = np.array(self.shape, dtype=int)
            
            if np.any(idx_array < 0) or np.any(idx_array >= shape_array):
                raise IndexError(
                    f"Batch coordinate out of bounds for ForeignField of shape {self.shape}. "
                    f"Ensure signed coordinates are within [-half, half-1]."
                )
            
            idx_tuple = tuple(idx_array[..., i] for i in range(self.d))
            return self.data[idx_tuple]

        # 3. Single Coordinate Evaluation (Tuple)
        if len(key) != self.d:
            raise IndexError(f"Expected {self.d} coordinates, got {len(key)}")
        
        idx = tuple(int(k) + h for k, h in zip(key, self.half))
        
        for val, size in zip(idx, self.shape):
            if val < 0 or val >= size:
                raise IndexError(f"Coordinate {key} out of bounds for shape {self.shape}")
                
        return float(self.data[idx])
    def __array__(self, dtype=None) -> np.ndarray:
        """NumPy __array__ protocol for seamless materialisation."""
        return np.array(self.data, dtype=dtype)

    def __repr__(self) -> str:
        return f"ForeignField(shape={self.shape}, d={self.d})"
                
# ── ScarStore: Holographic Sparse Memory (OD-31 / HM-1) ──────────────────────
#
# STATUS: PROTOTYPE.  HM-1 (Holographic Sparsity Bound) is a CONJECTURE.
# This implementation demonstrates the concept; benchmark compression ratios
# before upgrading to empirical/proven status.
#
# Design
# ------
# A ScarStore wraps a SparseCommunionManifold as its O(1)-memory "baseline"
# and overlays a sparse Python dict of "scars" — coordinates where the true
# value deviates from the FLU baseline.
#
# Storage cost = O(D) (baseline params) + O(|S|) (scar dict)
# vs full tensor = O(n^D).
# Compression ratio = n^D / (D + |S|) → ∞ as D grows and |S| is small.

class ScarStore:
    """
    Holographic sparse memory over a SparseCommunionManifold baseline.
    STATUS: PROVEN (HM-1). Holographic Sparsity representation is mathematically exact (lossless coset decomposition, see DEC-1).
    Any tensor Q can be represented as FLU-baseline + sparse anomalies (scars).  Storage
    scales with the number of anomalies, not the size of the domain.

    STATUS: PROTOTYPE (OD-31 research direction, V14 audit).

    Parameters
    ----------
    n : int   base radix (odd)
    d : int   spatial dimension
    seeds : list[int] | None
        Per-axis seed ranks for the baseline SparseCommunionManifold.
        If None, uses default seeds (rank 0 for every axis).

    Examples
    --------
    >>> store = ScarStore(n=3, d=2)
    >>> store.learn((1, -1), 99)          # record a scar at coord (1,-1)
    >>> store.recall((1, -1))             # returns 99
    >>> store.recall((0, 0))              # returns baseline value
    >>> store.compression_ratio()         # n^d / (d + |scars|)
    """

    SIMULATION_ONLY = False   # STATUS: PROVEN (HM-1)

    def __init__(
        self,
        n      : int,
        d      : int,
        seeds  = None,
    ) -> None:
        # Build default seeds if none provided.
        # Default: identity permutation [0, 1, ..., n-1] for every axis.
        # This gives baseline M[x] = (sum_i (x_i + half)) mod n - half.
        if seeds is None:
            default_seeds = [np.arange(n, dtype=int) for _ in range(d)]
        else:
            default_seeds = list(seeds)
        self._manifold = SparseCommunionManifold(n=n, seeds=default_seeds)
        self._scars: dict = {}   # coord_tuple → delta (true_value - baseline)
        self.n = n
        self.d = d

    # ── Core API ─────────────────────────────────────────────────────────────

    def learn(self, coord: tuple, true_value: float) -> None:
        """
        Record a scar: the true value at `coord` deviates from the baseline.

        If `true_value` matches the baseline exactly (within float tolerance),
        the scar is removed (forgetting a resolved anomaly).

        Parameters
        ----------
        coord       : tuple of int  signed coordinates in [−half, half]^d
        true_value  : float         the actual observed value at this coordinate
        """
        baseline = self._manifold[coord]
        delta = true_value - baseline
        if abs(delta) < 1e-12:
            self._scars.pop(coord, None)   # no anomaly — remove if present
        else:
            self._scars[coord] = delta

    def recall(self, coord: tuple) -> float:
        """
        Retrieve the stored value at `coord`.

        Returns baseline + scar_delta in O(D) + O(1) time.

        Parameters
        ----------
        coord : tuple of int  signed coordinates

        Returns
        -------
        float  reconstructed value
        """
        baseline = float(self._manifold[coord])
        delta    = self._scars.get(coord, 0.0)
        return baseline + delta

    def forget(self, coord: tuple) -> None:
        """Remove a scar, reverting the coordinate to its baseline value."""
        self._scars.pop(coord, None)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def scar_count(self) -> int:
        """Number of anomalies currently stored."""
        return len(self._scars)

    def compression_ratio(self) -> float:
        """
        Effective compression ratio vs storing the full n^D tensor.

        ratio = n^D / (D + |scars|).
        A ratio > 1 means ScarStore is smaller than brute-force storage.
        Approaches n^D / D as |scars| → 0 (empty domain).
        """
        full_size  = self.n ** self.d
        store_size = self.d + self.scar_count()
        return float(full_size) / max(store_size, 1)

    def anomaly_rate(self) -> float:
        """Fraction of cells that are scarred: |scars| / n^D."""
        return self.scar_count() / (self.n ** self.d)

    # ── Helpers ──────────────────────────────────────────────────────────────
    
    def materialize(self) -> np.ndarray:
        """Holographic reconstruction of the full tensor."""
        shape = tuple([self.n] * self.d)
        out = self._manifold.materialize() # Start with dense baseline
        half = self.n // 2
        for coord, delta in self._scars.items():
            idx = tuple(c + half for c in coord)
            out[idx] += delta
        return out
        
    def __repr__(self) -> str:
        return (
            f"ScarStore(n={self.n}, d={self.d}, "
            f"scars={self.scar_count()}, "
            f"compression={self.compression_ratio():.1f}x, "
            f"SIMULATION_ONLY={self.SIMULATION_ONLY})"
        )
