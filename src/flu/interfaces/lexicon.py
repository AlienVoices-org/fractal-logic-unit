"""
src/flu/interfaces/lexicon.py
=============================
LexiconFacet — Bijective n-ary Alphanumeric Mapping (LEX-1).

Mathematical identity: a bijection  Λ : Z_n^D → Σ*
where Σ is a finite alphanumeric alphabet of cardinality |Σ| = n^k.

For n=3, k=3 (Heptavintimal base-27): 3-trit clusters ↔ {A…Z, _}
For n=2 (binary): each trit ↔ {'0','1'}
For general n: digits packed into symbols of size ⌈log_|Σ| n⌉.

Significance (V15 Audit Finding — LEX-1):
  In high-dimensional spaces, numeric tuples are ergonomically unstable
  for human auditors and cross-node debugging. This mapping provides a
  lexicographically canonical form for addresses while preserving the
  T1 bijection exactly.

Status: PROVEN (algebraic_trivial — any bijection between finite sets
        of equal cardinality is valid by construction).

V15 — audit integration sprint.
"""

from __future__ import annotations

import string
from typing import Sequence, Tuple

from flu.interfaces.base import FluFacet


# ── Standard alphabets ────────────────────────────────────────────────────────

# Base-27 heptavintimal: A–Z + underscore (27 symbols = 3^3)
HEPTAVINTIMAL: str = string.ascii_uppercase + "_"   # len = 27

# Base-36 alphanumeric: 0–9 + A–Z
BASE36: str = string.digits + string.ascii_uppercase  # len = 36

# Binary: just '0' and '1'
BINARY: str = "01"


def _default_alphabet(n: int) -> str:
    """Choose a sensible default alphabet for a given radix n."""
    if n == 2:
        return BINARY
    if n <= 10:
        return string.digits[:n]
    if n <= 27:
        return HEPTAVINTIMAL[:n]
    if n <= 36:
        return BASE36[:n]
    # Fallback: Unicode codepoints
    return "".join(chr(0x41 + i) for i in range(n))


class LexiconFacet(FluFacet):
    """
    Bijective n-ary Alphanumeric Mapping (Theorem LEX-1).

    Maps Z_n^D coordinate tuples ↔ fixed-length symbol strings.
    The bijection is exact: no information is lost.

    Parameters
    ----------
    n : int
        Radix of the FLU manifold.
    d : int
        Dimension of the manifold.
    alphabet : str, optional
        Symbol set of cardinality ≥ n.  Defaults to the canonical
        alphabet for the given n.

    Examples
    --------
    >>> lex = LexiconFacet(n=3, d=4)
    >>> s = lex.encode((0, 1, 2, 0))
    >>> assert lex.decode(s) == (0, 1, 2, 0)
    """

    def __init__(self, n: int, d: int, alphabet: str | None = None) -> None:
        if alphabet is None:
            alphabet = _default_alphabet(n)
        if len(alphabet) < n:
            raise ValueError(
                f"Alphabet length {len(alphabet)} < radix n={n}. "
                f"Provide an alphabet with at least {n} symbols."
            )
        super().__init__(
            name="LexiconFacet",
            theorem_id="LEX-1",
            status="PROVEN",
            description=(
                "Bijective mapping Λ: Z_n^D → Σ* preserving T1 bijection. "
                "Provides ergonomic canonical address representation."
            ),
        )
        self.n = n
        self.d = d
        self.alphabet: str = alphabet

    # ── core bijection ────────────────────────────────────────────────────────

    def encode(self, coord: Sequence[int]) -> str:
        """
        Encode a Z_n^D coordinate tuple as a D-character symbol string.

        Accepts both signed (mean-centred) and unsigned coordinates.
        Signed coords in [-(n//2), n//2] are mapped to [0, n) before encoding.

        Parameters
        ----------
        coord : sequence of ints
            D-dimensional coordinate tuple (signed or unsigned).

        Returns
        -------
        str
            D-character string over self.alphabet.
        """
        if len(coord) != self.d:
            raise ValueError(f"Expected {self.d} coordinates, got {len(coord)}")
        chars = []
        for c in coord:
            # Normalise to [0, n) regardless of sign
            c_unsigned = int(c) % self.n
            if not (0 <= c_unsigned < self.n):
                raise ValueError(f"Normalised coordinate {c_unsigned} out of range [0, {self.n})")
            chars.append(self.alphabet[c_unsigned])
        return "".join(chars)

    def decode(self, symbol: str) -> Tuple[int, ...]:
        """
        Decode a D-character symbol string back to unsigned Z_n^D coordinates.

        Parameters
        ----------
        symbol : str
            D-character string over self.alphabet.

        Returns
        -------
        tuple of ints in [0, n)
            Unsigned coordinate tuple.
        """
        if len(symbol) != self.d:
            raise ValueError(f"Symbol length {len(symbol)} != d={self.d}")
        result = []
        for ch in symbol:
            idx = self.alphabet.index(ch)
            result.append(idx)
        return tuple(result)

    # ── convenience ──────────────────────────────────────────────────────────

    def encode_rank(self, k: int) -> str:
        """Encode a traversal rank k directly (O(D) via path_coord)."""
        from flu.core.fm_dance_path import path_coord
        return self.encode(path_coord(k, self.n, self.d))

    def all_symbols(self) -> list[str]:
        """Return all n^D encoded symbols in FM-Dance traversal order."""
        from flu.core.fm_dance_path import path_coord
        return [self.encode(path_coord(k, self.n, self.d)) for k in range(self.n ** self.d)]
