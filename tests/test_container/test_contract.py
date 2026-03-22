"""
Tests for flu.container.contract — UKMC Contract.

Key test: freeze() isolation — freezing one instance must NOT freeze others.
"""

import pytest
from flu.container.contract import UKMCContract


class TestIdentityHash:
    def test_deterministic(self):
        c = UKMCContract(logos={"x": 1}, omega=1.0, phi={"p": 0})
        h1 = c.identity_hash()
        h2 = c.identity_hash()
        assert h1 == h2

    def test_identical_contracts_equal_hashes(self):
        c1 = UKMCContract(logos={"x": 1}, omega=1.0, phi={"p": 0})
        c2 = UKMCContract(logos={"x": 1}, omega=1.0, phi={"p": 0})
        assert c1.identity_hash() == c2.identity_hash()

    def test_tau_does_not_affect_hash(self):
        c1 = UKMCContract(tau=0, logos={"x": 1}, omega=1.0, phi={})
        c2 = UKMCContract(tau=99, logos={"x": 1}, omega=1.0, phi={})
        assert c1.identity_hash() == c2.identity_hash()

    def test_delta_does_not_affect_hash(self):
        c = UKMCContract(logos={"x": 1}, omega=1.0, phi={})
        h_before = c.identity_hash()
        c.add_scar("REINFORCE", 0.5, (0, 0), "test scar")
        h_after  = c.identity_hash()
        assert h_before == h_after

    def test_logos_change_changes_hash(self):
        c1 = UKMCContract(logos={"x": 1}, omega=1.0, phi={})
        c2 = UKMCContract(logos={"x": 2}, omega=1.0, phi={})
        assert c1.identity_hash() != c2.identity_hash()

    def test_omega_change_changes_hash(self):
        c1 = UKMCContract(logos={}, omega=1.0, phi={})
        c2 = UKMCContract(logos={}, omega=2.0, phi={})
        assert c1.identity_hash() != c2.identity_hash()


class TestFreeze:
    def test_freeze_blocks_identity_fields(self):
        """After freeze(), mutating Λ/Ω/Φ must raise AttributeError."""
        c = UKMCContract(logos={"x": 1}, omega=1.0, phi={"p": 0})
        c.freeze()

        with pytest.raises(AttributeError):
            c.logos = {"x": 99}
        with pytest.raises(AttributeError):
            c.omega = 2.0
        with pytest.raises(AttributeError):
            c.phi = {"p": 1}

    def test_freeze_allows_mutable_fields(self):
        """After freeze(), τ, Δ, ⊗ must still be mutable."""
        c = UKMCContract(logos={}, omega=1.0, phi={})
        c.freeze()

        c.tau   = 5         # must not raise
        c.port  = {"k": 1}  # must not raise
        c.add_scar("WEAKEN", -0.1, (0,), "post-freeze scar")  # must not raise
        assert c.tau == 5
        assert len(c.delta) == 1

    def test_freeze_is_instance_level(self):
        """
        freeze() isolation regression:
        Freezing contract A must NOT freeze contract B.
        The old implementation mutated the class, not the instance.
        """
        c_a = UKMCContract(logos={"id": "A"}, omega=1.0, phi={})
        c_b = UKMCContract(logos={"id": "B"}, omega=2.0, phi={})

        c_a.freeze()
        assert c_a.is_frozen
        assert not c_b.is_frozen   # <-- the regression check

        # c_b must still be mutable
        c_b.omega = 3.0
        assert c_b.omega == 3.0

    def test_freeze_is_idempotent(self):
        c = UKMCContract()
        c.freeze()
        c.freeze()   # second call must not raise
        assert c.is_frozen


class TestScars:
    def test_reinforce_scar(self):
        c = UKMCContract()
        c.add_scar("REINFORCE", 0.9, (1, 2), "strong learning")
        assert len(c.delta) == 1
        assert c.delta[0]["type"] == "REINFORCE"

    def test_neutral_requires_extras(self):
        c = UKMCContract()
        with pytest.raises(ValueError, match="equality_candidates"):
            c.add_scar("NEUTRAL", 0.0, (0,), "paradox")

    def test_neutral_with_all_fields(self):
        c = UKMCContract()
        c.add_scar(
            "NEUTRAL", 0.0, (0,), "branch",
            equality_candidates=["A", "B"],
            paradox_description="simultaneous truths",
            branch_gnosis=0.5,
        )
        assert c.delta[0]["paradox_description"] == "simultaneous truths"

    def test_unknown_scar_type_raises(self):
        c = UKMCContract()
        with pytest.raises(ValueError, match="Unknown scar type"):
            c.add_scar("INVALID", 0.5, (0,), "bad type")
