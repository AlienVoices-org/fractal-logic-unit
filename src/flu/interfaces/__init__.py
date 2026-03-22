"""
flu.interfaces — V15 Bridge Facets
====================================
Exposes classical mathematical structures (Hilbert curves, DEC operators,
APN seed reservoirs, integrity sondes, Gray codes, Hadamard matrices,
experimental design, neural initialisation, digital nets) as FluFacet
components anchored to the FLU theorem registry.

Public exports
--------------
FluFacet                  — abstract base (all facets inherit from this)
LexiconFacet              — LEX-1    PROVEN         bijective n-ary alphanumeric mapping
IntegrityFacet            — INT-1    PROVEN         O(1) L1 conservative-law sonde
GeneticFacet              — GEN-1    PROVEN         SHA-256 hashed APN seed reservoirs
InvarianceFacet           — INV-1    PROVEN         odd/even branch structural isomorphism
HadamardFacet             — HAD-1    PROVEN         Sylvester-Hadamard via bit-masked Communion
CryptoFacet               — CRYPTO-1 PROVEN         APN prime-field structural immunity
HilbertFacet              — HIL-1    RETIRED        ⚠️ DEPRECATED — FM-Dance+RotationHub (n=2 contradiction)
CurveFacet                — HIL-1    RESEARCH       HIL-1 follow up: FM-Dance + new RotationHub for Peano like space filling
CohomologyFacet           — DEC-1    PROVEN         ScarStore = coset decomp of C⁰/SCM (proven V15.1.2)
GrayCodeFacet             — T8       PROVEN         FM-Dance as n-ary Gray code
DesignFacet               — T3       DESIGN_INTENT  Latin hypercube experimental design
NeuralFacet               — PFNT-2   DESIGN_INTENT  bias-free neural weight initialisation
FractalNetCorputFacet     — FMD-NET  PROVEN         van der Corput digital net (C_m=I, control)
FractalNetKineticFacet    — T9       PROVEN         FM-Dance kinetic digital net (C_m=T, Faure family)

V15 — audit integration sprint.
V15.1.2 — DEC-1 PROVEN (ScarStore coset decomposition via Künneth + HM-1).
V15.1.3 — HIL-1 RETIRED (n=2 primary case self-contradicts odd-n requirement).
"""

from flu.interfaces.base import FluFacet, FacetInfo
from flu.interfaces.lexicon import LexiconFacet
from flu.interfaces.integrity import IntegrityFacet
from flu.interfaces.genetic import GeneticFacet, SeedRecord
from flu.interfaces.invariance import InvarianceFacet
from flu.interfaces.hadamard import HadamardFacet
from flu.interfaces.crypto import CryptoFacet
from flu.interfaces.curves import CurveFacet
from flu.interfaces.hilbert import HilbertFacet, RotationHub
from flu.interfaces.cohomology import CohomologyFacet
from flu.interfaces.gray_code import GrayCodeFacet, binary_gray_encode, binary_gray_decode
from flu.interfaces.design import DesignFacet
from flu.interfaces.neural import NeuralFacet
from flu.interfaces.digital_net import FractalNetCorputFacet, FractalNetKineticFacet

__all__ = [
    "FluFacet",
    "FacetInfo",
    "LexiconFacet",
    "IntegrityFacet",
    "GeneticFacet",
    "SeedRecord",
    "InvarianceFacet",
    "HadamardFacet",
    "CryptoFacet",
    "HilbertFacet",
    "RotationHub",
    "CohomologyFacet",
    "GrayCodeFacet",
    "binary_gray_encode",
    "binary_gray_decode",
    "DesignFacet",
    "NeuralFacet",
    "FractalNetCorputFacet",
    "FractalNetKineticFacet",
]
