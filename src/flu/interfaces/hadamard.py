"""
src/flu/interfaces/hadamard.py
================================
HadamardFacet — Sylvester-Hadamard Matrix Generator via Communion (HAD-1).

STATUS: PROVEN (HAD-1, algebraic_and_computational)

MATHEMATICAL BASIS
------------------
Communion with bit-masked parametrised seeds generates Sylvester-Hadamard
matrices of order N = 2^D:

    Seeds:   π_a(x) = k_a ∧ x   (bit-masked identity, parametrised by row k)
    Fold:    C_k(x) = ⊕_a (k_a ∧ x_a) = k · x  (mod 2)   [dot-product mod 2]
    Row:     H_k(x) = (−1)^{C_k(x)}              [bipolar map]

The resulting matrix H satisfies H @ H.T = N · I exactly.
H is the character table of the elementary abelian 2-group Z_2^D.

IMPORTANT DISTINCTION (V15 audit correction)
---------------------------------------------
Earlier attempts used *static* identity seeds [0,1] for all axes and folded
via XOR — that maps to (−1)^{parity(x)}, not the required dot-product.
The corrected proof uses *parametrised* seeds (different seed per row k).
See flu/applications/hadamard.py for the full derivation.

SCOPE NOTE
-----------
Proves the 2^D (Sylvester) subfamily only.  Generalising to arbitrary 4k
orders is an open research direction.

V15 — audit integration sprint.
"""

from __future__ import annotations

import numpy as np

from flu.interfaces.base import FluFacet
from flu.applications.hadamard import HadamardGenerator


class HadamardFacet(FluFacet):
    """
    Bridge facet exposing Sylvester-Hadamard generation via parametrised
    XOR-Communion (HAD-1 PROVEN).

    Parameters
    ----------
    d : int
        Depth parameter.  The generated matrix has order N = 2^d.

    Examples
    --------
    >>> hf = HadamardFacet(d=4)
    >>> H = hf.generate()
    >>> H.shape
    (16, 16)
    >>> import numpy as np
    >>> np.array_equal(H @ H.T, 16 * np.eye(16, dtype=int))
    True
    >>> hf.info().theorem_id
    'HAD-1'
    """

    def __init__(self, d: int) -> None:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        super().__init__(
            name="HadamardFacet",
            theorem_id="HAD-1",
            status="PROVEN",
            description=(
                "Sylvester-Hadamard matrix generation via bit-masked "
                "parametrised Communion seeds. H @ H.T = N·I exactly."
            ),
        )
        self.d = d
        self._gen = HadamardGenerator()

    def generate(self) -> np.ndarray:
        """
        Generate the Sylvester-Hadamard matrix of order N = 2^d.

        Returns
        -------
        ndarray of shape (2^d, 2^d) with integer entries in {+1, -1}.
        """
        return self._gen.generate(self.d)

    def generate_row(self, k: int) -> np.ndarray:
        """
        Generate row k of the Hadamard matrix in O(N) time.

        Parameters
        ----------
        k : int
            Row index in [0, 2^d).

        Returns
        -------
        ndarray of shape (2^d,) with entries in {+1, -1}.
        """
        return self._gen.generate_row(k, self.d)

    def verify(self) -> bool:
        """Verify H @ H.T == N·I.  Returns True iff orthogonal."""
        return self._gen.verify(self.d)

    def __repr__(self) -> str:
        return f"HadamardFacet(d={self.d}, N={2**self.d}, theorem={self._theorem_id!r})"
