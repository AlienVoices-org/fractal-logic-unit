"""
src/flu/interfaces/neural.py
=============================
NeuralFacet — Bias-Free Neural Weight Initialisation Bridge (DESIGN_INTENT).

STATUS: DESIGN_INTENT (thin bridge wrapper; core guarantees PROVEN)

MATHEMATICAL GUARANTEES (inherited from core)
---------------------------------------------
  - Zero global mean (odd n, signed=True): PROVEN (PFNT-2)
  - Full coverage of digit set per slice: PROVEN (T3)
  - Latin property of each layer weight tensor: PROVEN (T3 / even-n)

DESIGN INTENT STATUS
--------------------
This facet is "design intent": it surfaces the proven zero-bias and
coverage guarantees via the FluFacet interface. The independence of
layers across different factoradic seeds is DESIGN INTENT (not formally
proven for correlation structure).

V15 — audit integration sprint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flu.interfaces.base import FluFacet


class NeuralFacet(FluFacet):
    """
    Bias-Free Neural Weight Initialisation bridge facet.

    Wraps flu.applications.neural.FLUInitializer and
    exposes it through the FluFacet interface.

    Parameters
    ----------
    n : int
        Base radix for the FLU initialiser.
    d : int
        Spatial dimension (controls tensor shape).
    signed : bool
        If True (default), use signed balanced representation.

    Examples
    --------
    >>> nf = NeuralFacet(n=5, d=3)
    >>> weights = nf.init_layer(shape=(5, 5, 5))
    >>> abs(weights.mean()) < 0.2  # near zero mean
    True
    >>> nf.info().theorem_id
    'PFNT-2'
    """

    def __init__(self, n: int, d: int, signed: bool = True) -> None:
        super().__init__(
            name="NeuralFacet",
            theorem_id="PFNT-2",
            status="DESIGN_INTENT",
            description=(
                "Bias-free weight initialisation using FLU Latin hyperprisms. "
                "Zero global mean (PFNT-2 PROVEN). "
                "Full-coverage digit set (T3 PROVEN)."
            ),
        )
        self.n = n
        self.d = d
        self.signed = signed

    def init_layer(self, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        Generate a weight tensor with zero-bias FLU initialisation.

        Parameters
        ----------
        shape : tuple of int, optional
            Desired output shape. If None, returns the full n^d hyperprism.
            If provided, the total size (prod(shape)) must be ≤ n^d.

        Returns
        -------
        ndarray
            Weight array with zero mean (odd n).

        Raises
        ------
        ValueError
            If prod(shape) > n^d. Tiling is mathematically forbidden as it 
            destroys the PROVEN Latin (T3) and Mean-Zero (PFNT-2) invariants.
        """
        from flu.applications.neural import FLUInitializer
        
        # 1. Instantiate correctly (only takes `signed`)
        init = FLUInitializer(signed=self.signed)
        
        # 2. Generate the base tensor using the Facet's specific n and d
        base_shape = tuple([self.n] * self.d)
        weights = init.weights(base_shape)
        
        # 3. Handle custom reshaping/slicing safely
        if shape is not None:
            flat = weights.flatten()
            target = int(np.prod(shape))
            
            # The Invariant Guardrail
            if target > flat.size:
                raise ValueError(
                    f"Requested shape {shape} (size {target}) exceeds the Facet's "
                    f"manifold capacity (n={self.n}, d={self.d}, size={flat.size}). "
                    f"Tiling is forbidden as it destroys the PROVEN Latin (T3) "
                    f"and Mean-Zero (PFNT-2) invariants. Increase n or d."
                )
            
            # Exact match: Invariants perfectly preserved
            if target == flat.size:
                return flat.reshape(shape)
                
            # Cropping: Invariants degrade to 'near-zero' and 'partial Latin'
            # This is permitted for engineering practicality, but logged as DESIGN_INTENT
            return flat[:target].reshape(shape)
            
        return weights
        
    def verify(self) -> Dict[str, Any]:
        """
        Generate and verify the initialisation properties.

        Returns
        -------
        dict with keys: 'mean', 'zero_mean_ok', 'latin_ok'.
        """
        weights = self.init_layer()
        from flu.utils.verification import check_latin
        latin_ok = bool(check_latin(weights, self.n, signed=self.signed))
        mean_val = float(weights.mean())
        return {
            "mean": mean_val,
            "zero_mean_ok": abs(mean_val) < 1e-9,
            "latin_ok": latin_ok,
        }

    def __repr__(self) -> str:
        return (f"NeuralFacet(n={self.n}, d={self.d}, "
                f"theorem={self._theorem_id!r})")
