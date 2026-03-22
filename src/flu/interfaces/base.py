"""
src/flu/interfaces/base.py
==========================
FluFacet — abstract base for all V15 interface facets.

A *facet* is a bridge module that exposes a classical mathematical object
(Hilbert curve, DEC operators, APN seed table …) through a common interface
anchored to its FLU conjecture / theorem ID.

Every concrete facet MUST set:
  - name        : short display name  (e.g. "LexiconFacet")
  - theorem_id  : registry key         (e.g. "LEX-1")

V15 — introduced in V15.0 audit integration sprint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FacetInfo:
    """Metadata record returned by FluFacet.info()."""
    name: str
    theorem_id: str
    status: str          # "PROVEN" | "CONJECTURE" | "DESIGN_INTENT"
    description: str
    version_added: str = "15.0.0"


class FluFacet(ABC):
    """
    Abstract base class for all FLU interface facets (V15+).

    Subclasses expose a specific mathematical structure as a usable
    interface component while staying anchored to the theorem registry.
    """

    def __init__(self, name: str, theorem_id: str,
                 status: str = "CONJECTURE",
                 description: str = "") -> None:
        self._name = name
        self._theorem_id = theorem_id
        self._status = status
        self._description = description

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def theorem_id(self) -> str:
        return self._theorem_id

    @property
    def status(self) -> str:
        return self._status

    def info(self) -> FacetInfo:
        """Return a FacetInfo record for this facet."""
        return FacetInfo(
            name=self._name,
            theorem_id=self._theorem_id,
            status=self._status,
            description=self._description,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, id={self._theorem_id!r}, status={self._status!r})"
