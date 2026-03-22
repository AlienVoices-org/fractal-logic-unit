"""
flu/container/communion.py
==========================
Communion Operator ⊗_φ — algebraic container fusion.

Single Responsibility: combine two hyperprisms into a higher-dimensional
structure.  No application logic, no simulation, no network code.

THEOREM (Communion Closure), STATUS: PROVEN (conditional on φ-associativity)
See theory.py for formal statement.

Design choices from audit
─────────────────────────
1. `CommunionEngine` is a pure algebraic engine (this file).
2. `QuantumGravitySimulator` lives in applications/quantum.py — it is a
   domain application, not algebraic infrastructure.
3. `MeshCommunionNetwork` belongs in a separate networking layer (future).

φ (phi) must be associative.  The constructor validates this with a
3-element spot-check.  Non-associative φ raises ValueError.

Dependencies: flu.core.fm_dance, numpy.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Optional, Union

import numpy as np


# ── Supported built-in φ operators ───────────────────────────────────────────

_BUILTIN_PHI: Dict[str, Callable] = {
    "add"     : lambda a, b: a + b,
    "xor"     : lambda a, b: int(a) ^ int(b),
    "multiply": lambda a, b: a * b,
}

# Associativity annotation for built-ins (all True)
_PHI_ASSOCIATIVE = {"add", "xor", "multiply"}


def _check_associativity(phi: Callable, samples: int = 20) -> bool:
    """
    Probabilistic spot-check: (a ⊗ b) ⊗ c == a ⊗ (b ⊗ c) for random triples.
    STATUS: DESIGN INTENT — not a formal proof, but catches common failures.
    """
    rng = np.random.default_rng(42)
    for _ in range(samples):
        vals = rng.integers(-10, 10, 3)
        a, b, c = int(vals[0]), int(vals[1]), int(vals[2])
        try:
            lhs = phi(phi(a, b), c)
            rhs = phi(a, phi(b, c))
        except Exception:
            return False
        if lhs != rhs:
            return False
    return True


# ── CommunionEngine ───────────────────────────────────────────────────────────

class CommunionEngine:
    """
    Communion Operator ⊗_φ: algebraic outer fusion of two hyperprisms.

    Usage
    -----
    >>> eng = CommunionEngine(phi="add")
    >>> result = eng.commune(A, B)   # outer product

    Parameters
    ----------
    phi : str | Callable
        Recombination function.  Built-in strings: "add", "xor", "multiply".
        Custom callable must be associative (checked on construction).

    mode : str
        "outer"     — outer product; result shape = A.shape + B.shape  (default)
        "direct"    — direct sum along a new dimension (requires same n)
        "kronecker" — Kronecker product (flattens both first)

    THEOREM (Communion Closure), STATUS: PROVEN (conditional):
        If phi is associative, the result array has the Latin property
        in every axis-aligned 1-D slice.
    """

    def __init__(
        self,
        phi : Union[str, Callable] = "add",
        mode: str                   = "outer",
    ) -> None:
        if isinstance(phi, str):
            if phi not in _BUILTIN_PHI:
                raise ValueError(
                    f"Unknown phi '{phi}'. Built-ins: {list(_BUILTIN_PHI)}"
                )
            self._phi  = _BUILTIN_PHI[phi]
            self._assoc_verified = True
        else:
            if not callable(phi):
                raise TypeError("phi must be a string or callable")
            if not _check_associativity(phi):
                raise ValueError(
                    "Custom phi failed associativity check.  "
                    "⊗_φ requires associative φ (see theory.py Theorem 5)."
                )
            self._phi            = phi
            self._assoc_verified = True

        if mode not in ("outer", "direct", "kronecker"):
            raise ValueError(f"Unknown mode '{mode}'. Use outer|direct|kronecker.")
        self.mode = mode

    # ── Arithmetics ─────────────────────────────────────────────────────
    @staticmethod
    def simplify(left: Any, right: Any, op: Callable, symbol: str) -> Any:
        """
        The Optimizer: Canonicalise the arithmetic tree (OPER-1).
        Prunes 'Dead Branches' to prevent memory leaks and redundant O(D) calls.
        """
        # Resolve lazy imports
        from flu.container.sparse import SparseArithmeticManifold, ConstantManifold

        # Safely extract dimension metadata from whichever operand is a manifold
        n = getattr(left, 'n', getattr(right, 'n', 0))
        d = getattr(left, 'd', getattr(right, 'd', 0))

        # ── Rule 1: Destructive Identity (Multiply by Zero) ──
        if symbol == "⊗":
            if (isinstance(right, (int, float)) and right == 0) or \
               (isinstance(left, (int, float)) and left == 0):
                return ConstantManifold(0.0, n, d)

        # ── Rule 2: Zero Numerator (Division) ──
        if symbol == "⊘":
            if isinstance(left, (int, float)) and left == 0:
                return ConstantManifold(0.0, n, d)

        # ── Rule 3: Additive Identity (Addition/Subtraction) ──
        if symbol == "⊕":
            if isinstance(right, (int, float)) and right == 0: return left
            if isinstance(left, (int, float)) and left == 0: return right
        if symbol == "⊖":
            if isinstance(right, (int, float)) and right == 0: return left

        # ── Rule 4: Multiplicative Identity ──
        if symbol == "⊗":
            if isinstance(right, (int, float)) and right == 1: return left
            if isinstance(left, (int, float)) and left == 1: return right
        if symbol == "⊘":
            if isinstance(right, (int, float)) and right == 1: return left
            
        # No simplification possible: construct the lazy node
        return SparseArithmeticManifold(left, right, op, symbol)

    # ── Main interface ─────────────────────────────────────────────────────
    def commune(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute C = A ⊗_φ B.

        Parameters
        ----------
        A, B : np.ndarray  (same dtype preferred)

        Returns
        -------
        C : np.ndarray

        Raises
        ------
        ValueError  for direct mode with mismatched n.
        """
        if self.mode == "outer":
            return self._outer(A, B)
        elif self.mode == "direct":
            return self._direct(A, B)
        else:
            return self._kronecker(A, B)

    # ── Outer product ─────────────────────────────────────────────────────

    def _outer(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Outer product communion.
        result[i₁…i_dA, j₁…j_dB] = φ(A[i₁…i_dA], B[j₁…j_dB]).

        THEOREM (Outer Communion closes Latin property):
            If A is Latin (shape (n,)*dA) and B is Latin (shape (n,)*dB),
            then for φ=add the result has Latin property in every axis.
        STATUS: PROVEN for φ=add (extends Theorem 3 to product dimensions).
        """
        phi     = self._phi
        shape_C = A.shape + B.shape
        C       = np.empty(shape_C, dtype=A.dtype)

        for a_idx in np.ndindex(*A.shape):
            a_val = A[a_idx]
            for b_idx in np.ndindex(*B.shape):
                C[a_idx + b_idx] = phi(a_val, B[b_idx])

        return C

    # ── Direct sum ────────────────────────────────────────────────────────

    def _direct(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Direct sum communion (requires A and B to have the same n).
        Combines along a new leading dimension.
        """
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "Direct sum requires equal leading dimension. "
                f"Got {A.shape[0]} vs {B.shape[0]}."
            )
        phi     = self._phi
        dA, dB  = A.ndim, B.ndim
        n       = A.shape[0]
        shape_C = tuple([n] * (dA + dB))
        C       = np.zeros(shape_C, dtype=A.dtype)

        for a_idx in np.ndindex(*A.shape):
            for b_idx in np.ndindex(*B.shape):
                C[a_idx + b_idx] = phi(int(A[a_idx]), int(B[b_idx]))

        return C

    # ── Kronecker ─────────────────────────────────────────────────────────

    def _kronecker(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Kronecker communion: np.kron on flat arrays, reshaped to A+B dims.
        Useful for quantum state tensor products.
        """
        flat_A   = A.flatten()
        flat_B   = B.flatten()
        flat_C   = np.kron(flat_A, flat_B)
        shape_C  = A.shape + B.shape
        return flat_C.reshape(shape_C)

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"CommunionEngine(mode={self.mode!r}, assoc_verified={self._assoc_verified})"
