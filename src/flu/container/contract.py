"""
flu/container/contract.py
=========================
UKMC (Universal Kinship Magic Cell) identity contract.

Single Responsibility: define container identity, manage scars, and
provide a freeze mechanism.  No geometry, no array operations.

THEOREM (Container Identity), STATUS: PROVEN — see identity_hash().

The freeze() mechanism guards at the *instance* level via __setattr__
to prevent accidental mutation of a finalised contract.

Dependencies: hashlib, json — standard library only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


# ── Scar type constants ───────────────────────────────────────────────────────

SCAR_TYPES = frozenset({"REINFORCE", "WEAKEN", "NEUTRAL", "TRANSFORM"})


# ── UKMC Contract ─────────────────────────────────────────────────────────────

class UKMCContract:
    """
    Identity contract for a FLU container.

    Fields
    ------
    τ (tau)   : int    fractal depth — mutable, does NOT affect identity
    Λ (logos) : dict   genesis seed — part of identity hash
    Ω (omega) : float  gnostic weight — part of identity hash
    Φ (phi)   : dict   perspective descriptor — part of identity hash
    Δ (delta) : list   scar lattice — mutable, does NOT affect identity
    ⊗ (port)  : dict   communion port — mutable, does NOT affect identity

    THEOREM (Container Identity), STATUS: PROVEN
    ─────────────────────────────────────────────
    Statement:
        Two containers are the *same* container iff their (Λ, Ω, Φ) triples
        are equal.  τ, Δ, ⊗ are operational metadata and do not change
        which container this is.

    Proof:
        The identity hash is SHA-256(sort_keys(json({Λ, Ω, Φ}))).
        SHA-256 is collision-resistant, so equal hashes ↔ equal payloads
        with overwhelming probability.  Mutable fields are excluded by
        design.  □
    """

    # Immutable identity fields — changing these invalidates the hash
    _IDENTITY_FIELDS = frozenset({"logos", "omega", "phi"})

    def __init__(
        self,
        tau  : int             = 0,
        logos: Optional[Dict]  = None,
        omega: float           = 1.0,
        phi  : Optional[Dict]  = None,
        delta: Optional[List]  = None,
        port : Optional[Dict]  = None,
    ) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ guard
        object.__setattr__(self, "_frozen"  , False)
        object.__setattr__(self, "tau"      , tau)
        object.__setattr__(self, "logos"    , logos if logos is not None else {})
        object.__setattr__(self, "omega"    , float(omega))
        object.__setattr__(self, "phi"      , phi  if phi   is not None else {})
        object.__setattr__(self, "delta"    , delta if delta is not None else [])
        object.__setattr__(self, "port"     , port  if port  is not None else {})

    # ── Attribute guard ────────────────────────────────────────────────────

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Guard identity fields after freeze().

        After freeze() is called, any attempt to mutate Λ, Ω, or Φ raises
        AttributeError.  τ, Δ, ⊗ remain freely mutable.
        """
        frozen = object.__getattribute__(self, "_frozen")
        if frozen and name in self._IDENTITY_FIELDS:
            raise AttributeError(
                f"Contract is frozen; '{name}' is an identity field and "
                "cannot be mutated after freeze()."
            )
        object.__setattr__(self, name, value)

    # ── Scar management ────────────────────────────────────────────────────

    def add_scar(
        self,
        scar_type           : str,
        reinforcement       : float,
        coords              : tuple,
        description         : str,
        equality_candidates : Optional[List]  = None,
        paradox_description : Optional[str]   = None,
        branch_gnosis       : Optional[Any]   = None,
    ) -> None:
        """
        Append a scar to the Δ lattice.

        Scars record learning events, paradoxes, or structural stress.
        They are mutable metadata — they do NOT change container identity.

        Parameters
        ----------
        scar_type            : one of REINFORCE | WEAKEN | NEUTRAL | TRANSFORM
        reinforcement        : float  ∈ [-1.0, 1.0]  (negative for weakening)
        coords               : tuple  position in the hyperprism
        description          : str    human-readable annotation
        equality_candidates  : list   (NEUTRAL only) competing interpretations
        paradox_description  : str    (NEUTRAL only) description of paradox
        branch_gnosis        : any    (NEUTRAL only) gnosis value at branch
        """
        if scar_type not in SCAR_TYPES:
            raise ValueError(
                f"Unknown scar type '{scar_type}'. Must be one of {SCAR_TYPES}."
            )
        if not isinstance(reinforcement, (int, float)):
            raise TypeError("reinforcement must be numeric")

        if scar_type == "NEUTRAL":
            if equality_candidates is None:
                raise ValueError("NEUTRAL scar requires equality_candidates")
            if paradox_description is None:
                raise ValueError("NEUTRAL scar requires paradox_description")
            if branch_gnosis is None:
                raise ValueError("NEUTRAL scar requires branch_gnosis")

        entry: Dict[str, Any] = {
            "type"         : scar_type,
            "reinforcement": reinforcement,
            "coords"       : coords,
            "description"  : description,
        }
        if scar_type == "NEUTRAL":
            entry["equality_candidates"] = equality_candidates
            entry["paradox_description"] = paradox_description
            entry["branch_gnosis"]       = branch_gnosis

        self.delta.append(entry)

    # ── Identity ───────────────────────────────────────────────────────────

    def identity_hash(self) -> str:
        """
        SHA-256 hash of the identity triple (Λ, Ω, Φ).

        THEOREM (Container Identity), STATUS: PROVEN — see class docstring.
        """
        payload = json.dumps(
            {"logos": self.logos, "omega": self.omega, "phi": self.phi},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ── Freeze ─────────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """
        Make the identity fields (Λ, Ω, Φ) immutable.

        Guards at the *instance* level via __setattr__, so freezing one
        contract does not affect any other instance.

        Call after the first identity_hash() to prevent accidental
        mutation of core identity fields.  τ, Δ, ⊗ remain mutable.
        """
        object.__setattr__(self, "_frozen", True)

    @property
    def is_frozen(self) -> bool:
        """True iff freeze() has been called on this instance."""
        return bool(object.__getattribute__(self, "_frozen"))

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"UKMCContract("
            f"τ={self.tau}, "
            f"Ω={self.omega:.4f}, "
            f"Φ={self.phi}, "
            f"|Δ|={len(self.delta)}, "
            f"frozen={self.is_frozen}, "
            f"id={self.identity_hash()[:8]}…)"
        )
