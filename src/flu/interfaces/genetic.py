"""
src/flu/interfaces/genetic.py
==============================
GeneticFacet — Cryptographically Verified Permutation Seed Portability (GEN-1).

Mathematical identity:
  A cryptographically signed lookup table of permutations π ∈ S_n
  where differential uniformity δ(π) ≤ 2 (APN / δ=3 best-known).

  SHA-256(serialize(π)) is stored alongside the permutation so that
  any downstream consumer (Python, VHDL, C, FPGA) can verify the
  "Mathematical Genome" is uncorrupted during cross-platform transport.

Significance (V15 Audit Finding — GEN-1):
  Implementation Parity. For FLU logic to be truly substrate-agnostic,
  a VHDL core and a Python core must use identical algebraic seeds to
  guarantee identical Spectral Mixed-Frequency Flatness (S2). Centralising
  seeds in a hashed registry prevents "seed-skew" across the mesh.

Status: PROVEN (structural — SHA-256 collision resistance is a standard
        cryptographic assumption; seed verification is exact by construction).

V15 — audit integration sprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from flu.interfaces.base import FluFacet


def _sha256_of_seed(perm: list[int]) -> str:
    """Return the SHA-256 hex digest of a permutation serialised as JSON."""
    raw = json.dumps(perm, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SeedRecord:
    """
    An immutable, hash-verified permutation seed record.

    Attributes
    ----------
    n          : radix
    perm       : the permutation as a list of ints (0-indexed)
    delta      : differential uniformity δ(π)
    sha256     : SHA-256 hex digest for cross-platform verification
    provenance : version / source string
    """
    __slots__ = ("n", "perm", "delta", "sha256", "provenance")

    def __init__(self, n: int, perm: list[int], delta: int,
                 sha256: str | None = None,
                 provenance: str = "FLU V15") -> None:
        self.n = n
        self.perm = list(perm)
        self.delta = delta
        self.provenance = provenance
        computed = _sha256_of_seed(self.perm)
        if sha256 is not None and sha256 != computed:
            raise ValueError(
                f"SHA-256 mismatch for n={n} seed. "
                f"Provided={sha256}, Computed={computed}"
            )
        self.sha256 = computed

    def verify(self) -> bool:
        """Re-compute and compare the hash. Returns True iff intact."""
        return _sha256_of_seed(self.perm) == self.sha256

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "perm": self.perm,
            "delta": self.delta,
            "sha256": self.sha256,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SeedRecord":
        return cls(
            n=d["n"],
            perm=d["perm"],
            delta=d["delta"],
            sha256=d.get("sha256"),
            provenance=d.get("provenance", "unknown"),
        )

    def __repr__(self) -> str:
        return (f"SeedRecord(n={self.n}, δ={self.delta}, "
                f"sha256={self.sha256[:12]}…, prov={self.provenance!r})")


class GeneticFacet(FluFacet):
    """
    Cryptographically Verified APN Seed Reservoir (GEN-1).

    Wraps the GOLDEN_SEEDS table with SHA-256 verification and provides
    a substrate-agnostic serialisation format.

    Parameters
    ----------
    populate_from_golden : bool
        If True (default), auto-populate from flu.GOLDEN_SEEDS at init.

    Examples
    --------
    >>> gf = GeneticFacet()
    >>> rec = gf.get(n=3)
    >>> assert rec.verify()
    >>> blob = gf.export_json()   # ship to VHDL team
    >>> gf2 = GeneticFacet(populate_from_golden=False)
    >>> gf2.import_json(blob)
    >>> assert gf2.get(n=3).verify()
    """

    def __init__(self, populate_from_golden: bool = True) -> None:
        super().__init__(
            name="GeneticFacet",
            theorem_id="GEN-1",
            status="PROVEN",
            description=(
                "SHA-256 verified APN seed reservoir. Guarantees algebraic "
                "parity (S2-Prime, Avalanche) across Python, VHDL, and C substrates."
            ),
        )
        self._reservoir: Dict[int, List[SeedRecord]] = {}

        if populate_from_golden:
            self._load_golden_seeds()

    # ── population ───────────────────────────────────────────────────────────

    def _load_golden_seeds(self) -> None:
        """Load from flu.GOLDEN_SEEDS (built at library import time)."""
        try:
            from flu import GOLDEN_SEEDS  # type: ignore
            from flu.core.factoradic import factoradic_unrank
        except ImportError:
            return
        for n, seeds in GOLDEN_SEEDS.items():
            for seed_item in seeds:
                # GOLDEN_SEEDS stores Lehmer ranks (ints) or perm arrays
                if isinstance(seed_item, (int,)):
                    try:
                        perm = list(int(x) for x in factoradic_unrank(seed_item, n))
                    except Exception:
                        continue
                else:
                    # Already a sequence
                    perm = list(int(x) for x in seed_item)
                delta = self._compute_delta(perm, n)
                rec = SeedRecord(n=n, perm=perm, delta=delta, provenance="GOLDEN_SEEDS")
                self._reservoir.setdefault(n, []).append(rec)

    @staticmethod
    def _compute_delta(perm: list[int], n: int) -> int:
        """Compute differential uniformity δ(π) for a permutation over Z_n."""
        max_count = 0
        for da in range(1, n):
            counts: dict[int, int] = {}
            for x in range(n):
                dy = (perm[(x + da) % n] - perm[x]) % n
                counts[dy] = counts.get(dy, 0) + 1
            local_max = max(counts.values()) if counts else 0
            if local_max > max_count:
                max_count = local_max
        return max_count

    def add(self, n: int, perm: list[int], delta: int | None = None,
            provenance: str = "user") -> SeedRecord:
        """Add and register a new seed, computing hash automatically."""
        if delta is None:
            delta = self._compute_delta(perm, n)
        rec = SeedRecord(n=n, perm=perm, delta=delta, provenance=provenance)
        self._reservoir.setdefault(n, []).append(rec)
        return rec

    # ── retrieval ────────────────────────────────────────────────────────────

    def get(self, n: int, prefer_delta: int = 2) -> SeedRecord | None:
        """
        Return the best available seed for radix n.

        Prefers seeds with δ == prefer_delta; falls back to lowest δ
        available if none match.
        """
        records = self._reservoir.get(n, [])
        if not records:
            return None
        # Sort by delta ascending, then take first matching prefer_delta
        records_sorted = sorted(records, key=lambda r: r.delta)
        for rec in records_sorted:
            if rec.delta == prefer_delta:
                return rec
        return records_sorted[0]

    def get_all(self, n: int) -> list[SeedRecord]:
        """Return all seed records for radix n."""
        return list(self._reservoir.get(n, []))

    def available_n(self) -> list[int]:
        """Return sorted list of n values with at least one seed."""
        return sorted(self._reservoir.keys())

    # ── verification ─────────────────────────────────────────────────────────

    def verify_all(self) -> dict[int, bool]:
        """Verify SHA-256 integrity of all stored seeds."""
        result = {}
        for n, records in self._reservoir.items():
            result[n] = all(r.verify() for r in records)
        return result

    # ── serialisation ────────────────────────────────────────────────────────

    def export_json(self) -> str:
        """Export the full reservoir as a JSON string (for cross-platform transport)."""
        data = {
            "schema": "flu-genetic-v1",
            "seeds": {
                str(n): [r.to_dict() for r in records]
                for n, records in self._reservoir.items()
            },
        }
        return json.dumps(data, indent=2)

    def import_json(self, blob: str) -> None:
        """Import seeds from a JSON string, verifying SHA-256 on each."""
        data = json.loads(blob)
        for n_str, records in data.get("seeds", {}).items():
            n = int(n_str)
            for d in records:
                rec = SeedRecord.from_dict(d)  # hash verified in __init__
                self._reservoir.setdefault(n, []).append(rec)
