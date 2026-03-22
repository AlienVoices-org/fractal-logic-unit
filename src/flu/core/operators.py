"""
src/flu/core/operators.py
=========================
FLUOperator Base Class — The "Physics" Hook for external extensions.

STATUS: DESIGN INTENT (V16 API Foundation).

Provides the abstract protocol for all algebraic transformations within the
FLU ecosystem. External SRP packages (crypto, memory, simulation) inherit 
from FLUOperator to ensure interoperability with the Sparse Arithmetic Stack.

EPISTEMIC STATUS: SIMULATION ONLY.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

class FLUOperator(ABC):
    """
    Abstract Base Class for all FLU algebraic operators.
    
    An operator represents a "Natural Law" or "Physics" applied to the lattice.
    """
    def __init__(
        self, 
        theorem_id: str, 
        status: str = "DESIGN_INTENT",
        label: str = "abstract_op"
    ):
        self.theorem_id = theorem_id
        self.status = status
        self.label = label

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """
        Execute the operator logic.
        Must maintain O(D) or O(n) complexity to prevent manifold collapse.
        """
        pass

    def __repr__(self) -> str:
        return f"FLUOperator(id={self.theorem_id!r}, label={self.label!r}, status={self.status!r})"

# ── Reference Implementation of "Native Physics" ─────────────────────────────
class TMatrixOperator(FLUOperator):
    """
    The Discrete Integral (Prefix-Sum) Operator.
    Ref: T9, DISC-1 (PROVEN).
    """
    def __init__(self, n: int):
        super().__init__(theorem_id="T9", status="PROVEN", label="Integrate")
        self.n = n

    def __call__(self, digits: np.ndarray) -> np.ndarray:
        """x = T · a mod n (The FM-Dance Skew)."""
        res = np.cumsum(digits, axis=-1) % self.n
        res[..., 0] = (-digits[..., 0]) % self.n
        return res
        
class APNPermuteOperator(FLUOperator):
    """
    Non-linear Spectral Shatterer.
    Ref: DN2 (PARTIAL).
    """
    def __init__(self, n: int, seed: np.ndarray):
        super().__init__(theorem_id="DN2", status="PARTIAL", label="Permute")
        self.n = n
        self.seed = seed

    def __call__(self, x: Any) -> Any:
        """Apply APN permutation mapping."""
        return self.seed[x]

# ── The Rotation Hub Operator ──────────────────────────────────────────

class RotationHubOperator(FLUOperator):
    """
    Dynamically computes the H_D rotation state for a sub-cell.    
    T9 + HIL-2 Foundation.
    """
    def __init__(self, n: int, d: int, transition_rule: Callable) -> None:
        super().__init__(theorem_id="HIL-1", status="RESEARCH", label="Rotate")
        self.n = n
        self.d = d
        self.R = transition_rule  
        
        # Build the T matrix locally to strictly avoid cross-module import cycles
        self.T = np.tril(np.ones((d, d), dtype=int))
        self.T[0, 0] = -1

    def __call__(self, digits_hierarchy: List[np.ndarray]) -> np.ndarray:
        final_coord = np.zeros(self.d, dtype=float)
        current_omega = np.eye(self.d, dtype=int)
        half = self.n // 2
    
        for m, a_m in enumerate(digits_hierarchy):
            base_x = (self.T @ a_m) % self.n - half
            rotated_x = current_omega @ base_x
            final_coord += rotated_x / (self.n ** (m + 1))
            current_omega = current_omega @ self.R(a_m)
        
        return final_coord

# ── The "Hook" for external Physics ──────────────────────────────────────────

class ExternalPhysics(FLUOperator):
    """
    A generic wrapper for third-party SRP logic (e.g., flu-crypto).
    """
    def __init__(self, theorem_id: str, logic: callable, label: str = "External"):
        super().__init__(theorem_id=theorem_id, status="EXTERNAL_PROXY", label=label)
        self.logic = logic

    def __call__(self, x: Any) -> Any:
        return self.logic(x)
