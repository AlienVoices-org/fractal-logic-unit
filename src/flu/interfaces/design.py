"""
src/flu/interfaces/design.py
=============================
DesignFacet — Latin Hypercube Experimental Design Bridge (DESIGN_INTENT).

STATUS: DESIGN_INTENT (thin bridge wrapper; maths proven in core layer)

MATHEMATICAL GUARANTEES (inherited from core)
---------------------------------------------
  - Latin property: each symbol appears exactly once per axis-slice (T3, PROVEN)
  - Coverage guarantee: every value in {0,…,n-1} appears in each slice (T3)
  - Mean-centering (odd n, signed=True): global mean = 0 (PFNT-2, PROVEN)

DESIGN INTENT STATUS
--------------------
This facet itself is "design intent": it is a thin bridge that surfaces the
proven core guarantees via a clean FluFacet interface. The underlying
mathematical properties are proven; the facet wrapping is architectural.

V15 — audit integration sprint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from flu.interfaces.base import FluFacet


class DesignFacet(FluFacet):
    """
    Latin Hypercube Experimental Design bridge facet.

    Wraps flu.applications.design.ExperimentalDesign and
    exposes it through the FluFacet interface, with theorem provenance.

    Parameters
    ----------
    n_levels : int
        Number of levels (factor levels per run).
    d : int
        Number of dimensions (factors in the experiment).
    signed : bool
        If True (default), use balanced signed representation.

    Examples
    --------
    >>> df = DesignFacet(n_levels=5, d=4)
    >>> result = df.generate()
    >>> result.design.shape
    (5, 5, 5, 5)
    >>> df.info().theorem_id
    'T3'
    """

    def __init__(self, n_levels: int, d: int, signed: bool = True) -> None:
        super().__init__(
            name="DesignFacet",
            theorem_id="T3",
            status="DESIGN_INTENT",
            description=(
                "Latin hypercube experimental design. "
                "Thin bridge over ExperimentalDesign. "
                "Latin property and coverage guaranteed by T3 (PROVEN)."
            ),
        )
        self.n_levels = n_levels
        self.d = d
        self.signed = signed

    def generate(self) -> Any:
        """
        Generate a Latin hypercube design.

        Returns
        -------
        DesignResult
            A DesignResult object with .design (ndarray), .is_latin,
            .is_covered, .is_mean_centered attributes.
        """
        from flu.applications.design import ExperimentalDesign
        ed = ExperimentalDesign(
            n_levels=self.n_levels,
            d=self.d,
            signed=self.signed,
        )
        return ed.generate()

    def verify(self) -> Dict[str, bool]:
        """
        Generate and verify the Latin hypercube design.

        Returns
        -------
        dict with keys: 'is_latin', 'is_covered', 'is_mean_centered'.
        """
        result = self.generate()
        return {
            "is_latin": result.is_latin,
            "is_covered": result.is_covered,
            "is_mean_centered": result.is_mean_centered,
        }

    def __repr__(self) -> str:
        return (f"DesignFacet(n_levels={self.n_levels}, d={self.d}, "
                f"theorem={self._theorem_id!r})")
