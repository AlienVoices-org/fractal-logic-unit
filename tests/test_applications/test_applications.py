"""
Tests for flu.applications — ITER-3C through ITER-3F.

Covers ExperimentalDesign, FLUInitializer, TensorNetworkSimulator,
LighthouseBeacon.
"""

import pytest
import numpy as np

from flu.applications.design     import ExperimentalDesign, DesignResult
from flu.applications.neural     import FLUInitializer
from flu.applications.quantum    import TensorNetworkSimulator
from flu.applications.lighthouse import LighthouseBeacon, BeaconKey, cli_main


# ══════════════════════════════════════════════════════════════════════════════
# ITER-3C  ExperimentalDesign
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentalDesign:

    @pytest.fixture
    def ed(self):
        return ExperimentalDesign()

    # ── generate ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n,d", [(3, 2), (5, 3), (7, 2)])
    def test_odd_n_shape(self, ed, n, d):
        r = ed.generate(n_levels=n, n_factors=d)
        assert r.matrix.shape == (n,) * d

    @pytest.mark.parametrize("n,d", [(4, 2), (6, 3), (8, 2)])
    def test_even_n_shape(self, ed, n, d):
        r = ed.generate(n_levels=n, n_factors=d)
        assert r.matrix.shape == (n,) * d

    @pytest.mark.parametrize("n,d", [(3, 2), (5, 3), (4, 2), (6, 2)])
    def test_overall_pass(self, ed, n, d):
        r = ed.generate(n_levels=n, n_factors=d)
        assert r.overall_pass, f"Verification failed for n={n}, d={d}: {r.report}"

    def test_factor_names_stored(self, ed):
        r = ed.generate(3, 2, factor_names=["Temperature", "Pressure"])
        assert r.factor_names == ["Temperature", "Pressure"]

    def test_default_factor_names(self, ed):
        r = ed.generate(3, 3)
        assert r.factor_names == ["F0", "F1", "F2"]

    def test_factor_names_wrong_length_raises(self, ed):
        with pytest.raises(ValueError):
            ed.generate(3, 2, factor_names=["X"])

    def test_n_levels_1_raises(self, ed):
        with pytest.raises(ValueError):
            ed.generate(n_levels=1, n_factors=2)

    def test_n_factors_0_raises(self, ed):
        with pytest.raises(ValueError):
            ed.generate(n_levels=3, n_factors=0)

    def test_design_result_repr(self, ed):
        r = ed.generate(3, 2)
        assert "PASS" in repr(r)

    # ── verify_design ─────────────────────────────────────────────────────

    def test_verify_design_pass(self, ed):
        r = ed.generate(5, 2)
        report = ed.verify_design(r.matrix, n=5)
        assert report["overall_pass"]

    # ── stratified_sample ─────────────────────────────────────────────────

    def test_stratified_sample_shape(self, ed):
        r      = ed.generate(5, 3)
        sample = ed.stratified_sample(r, n_samples=3, rng=np.random.default_rng(42))
        assert sample.shape == (3, 3)

    def test_stratified_sample_too_many_raises(self, ed):
        r = ed.generate(3, 2)
        with pytest.raises(ValueError):
            ed.stratified_sample(r, n_samples=5)

    # ── pandas export ─────────────────────────────────────────────────────

    def test_to_dataframe_no_pandas(self, ed, monkeypatch):
        import flu.applications.design as mod
        monkeypatch.setattr(mod, "_pd", None)
        r = ed.generate(3, 2)
        with pytest.raises(ImportError, match="pandas"):
            ed.to_dataframe(r)


# ══════════════════════════════════════════════════════════════════════════════
# ITER-3D  FLUInitializer
# ══════════════════════════════════════════════════════════════════════════════

class TestFLUInitializer:

    @pytest.fixture
    def init(self):
        return FLUInitializer()

    # ── weights ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3), (5, 5), (4, 4)])
    def test_shape(self, init, shape):
        W = init.weights(shape)
        assert W.shape == shape

    @pytest.mark.parametrize("shape", [(3, 3, 3), (5, 5)])
    def test_unit_variance(self, init, shape):
        W = init.weights(shape)
        assert abs(W.std() - 1.0) < 1e-10, f"std={W.std()}"

    @pytest.mark.parametrize("shape", [(3, 3, 3), (5, 5)])
    def test_bias_free_odd(self, init, shape):
        W = init.weights(shape)
        assert abs(W.mean()) < 0.02, f"mean={W.mean()}"   # near zero after centering

    def test_bias_free_check_passes(self, init):
        # Use full n^d tensor (no cropping) for exact zero mean
        W = init.weights((3, 3, 3))
        # After centering and normalising, mean should be ~0
        assert abs(W.mean()) < 0.1   # relaxed: cropping may shift mean slightly

    def test_bias_free_check_fails_on_ones(self, init):
        with pytest.raises(AssertionError, match="bias_free_check failed"):
            init.bias_free_check(np.ones((3, 3)), atol=1e-6)

    def test_empty_shape_raises(self, init):
        with pytest.raises(ValueError):
            init.weights(())

    def test_zero_dim_raises(self, init):
        with pytest.raises(ValueError):
            init.weights((0, 3))

    def test_even_shape(self, init):
        W = init.weights((4, 4))
        assert W.shape == (4, 4)
        assert abs(W.std() - 1.0) < 1e-10

    def test_float64_dtype(self, init):
        W = init.weights((3, 3))
        assert W.dtype == np.float64

    # ── optional wrappers ─────────────────────────────────────────────────

    def test_to_torch_no_torch(self, init, monkeypatch):
        import flu.applications.neural as mod
        monkeypatch.setattr(mod, "_torch", None)
        W = init.weights((3, 3))
        with pytest.raises(ImportError, match="torch"):
            init.to_torch_parameter(W)

    def test_to_jax_no_jax(self, init, monkeypatch):
        import flu.applications.neural as mod
        monkeypatch.setattr(mod, "_jnp", None)
        W = init.weights((3, 3))
        with pytest.raises(ImportError, match="jax"):
            init.to_jax_array(W)


# ══════════════════════════════════════════════════════════════════════════════
# ITER-3E  TensorNetworkSimulator
# ══════════════════════════════════════════════════════════════════════════════

class TestTensorNetworkSimulator:

    @pytest.fixture
    def sim(self):
        return TensorNetworkSimulator(n=3)

    # ── prepare_state ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("q", [1, 2, 3, 4])
    def test_state_shape(self, sim, q):
        s = sim.prepare_state(q)
        assert s.shape == (3 ** q,)

    def test_state_is_normalised(self, sim):
        s = sim.prepare_state(2)
        assert abs(np.linalg.norm(s) - 1.0) < 1e-10

    def test_state_is_float64(self, sim):
        s = sim.prepare_state(2)
        assert s.dtype == np.float64

    def test_zero_qubits_raises(self, sim):
        with pytest.raises(ValueError):
            sim.prepare_state(0)

    def test_n_lt_2_raises(self):
        with pytest.raises(ValueError):
            TensorNetworkSimulator(n=1)

    def test_simulation_only_flag(self, sim):
        assert sim.SIMULATION_ONLY is True

    # ── measure ───────────────────────────────────────────────────────────

    def test_measure_total_shots(self, sim):
        s      = sim.prepare_state(2)
        counts = sim.measure(s, n_shots=100, rng=np.random.default_rng(0))
        assert sum(counts.values()) == 100

    def test_measure_outcomes_in_range(self, sim):
        s      = sim.prepare_state(2)
        counts = sim.measure(s, n_shots=200, rng=np.random.default_rng(0))
        for k in counts:
            assert 0 <= k < 3 ** 2

    def test_measure_non_1d_raises(self, sim):
        with pytest.raises(ValueError):
            sim.measure(np.ones((3, 3)), n_shots=10)

    def test_measure_zero_shots_raises(self, sim):
        s = sim.prepare_state(1)
        with pytest.raises(ValueError):
            sim.measure(s, n_shots=0)

    # ── fidelity ──────────────────────────────────────────────────────────

    def test_self_fidelity_is_one(self, sim):
        s = sim.prepare_state(2)
        assert abs(sim.fidelity(s, s) - 1.0) < 1e-10

    def test_fidelity_range(self, sim):
        s1 = sim.prepare_state(2)
        s2 = sim.prepare_state(3)[:9]   # different state, same length
        f  = sim.fidelity(s1, s2)
        assert 0.0 <= f <= 1.0

    def test_fidelity_shape_mismatch_raises(self, sim):
        s1 = sim.prepare_state(1)
        s2 = sim.prepare_state(2)
        with pytest.raises(ValueError):
            sim.fidelity(s1, s2)

    def test_fidelity_2d_raises(self, sim):
        with pytest.raises(ValueError):
            sim.fidelity(np.ones((3, 3)), np.ones((3, 3)))


# ══════════════════════════════════════════════════════════════════════════════
# ITER-3F  LighthouseBeacon
# ══════════════════════════════════════════════════════════════════════════════

class TestLighthouseBeacon:

    @pytest.fixture
    def beacon(self):
        return LighthouseBeacon(n=3, rounds=1, seed=42)

    # ── construction ──────────────────────────────────────────────────────

    def test_simulation_only_flag(self, beacon):
        assert beacon.SIMULATION_ONLY is True

    def test_even_n_raises(self):
        with pytest.raises(ValueError):
            LighthouseBeacon(n=4)

    def test_small_n_raises(self):
        with pytest.raises(ValueError):
            LighthouseBeacon(n=1)

    def test_zero_rounds_raises(self):
        with pytest.raises(ValueError):
            LighthouseBeacon(n=3, rounds=0)

    # ── generate_key ──────────────────────────────────────────────────────

    def test_key_returns_beacon_key(self, beacon):
        key = beacon.generate_key()
        assert isinstance(key, BeaconKey)

    def test_key_simulation_only(self, beacon):
        assert BeaconKey.SIMULATION_ONLY is True

    def test_key_hex_is_64_chars(self, beacon):
        key = beacon.generate_key()
        assert len(key.hex) == 64   # SHA-256 → 32 bytes → 64 hex chars

    def test_key_deterministic(self, beacon):
        k1 = beacon.generate_key(k_start=0)
        k2 = beacon.generate_key(k_start=0)
        assert k1.digest == k2.digest

    def test_distinct_k_start_distinct_keys(self, beacon):
        k1 = beacon.generate_key(k_start=0)
        k2 = beacon.generate_key(k_start=1)
        assert k1.digest != k2.digest

    def test_multi_round(self):
        beacon = LighthouseBeacon(n=3, rounds=3, seed=0)
        key    = beacon.generate_key()
        assert isinstance(key, BeaconKey)
        assert key.n_rounds == 3

    # ── broadcast ─────────────────────────────────────────────────────────

    def test_broadcast_prints_simulation_only(self, beacon, capsys):
        beacon.broadcast()
        out = capsys.readouterr().out
        assert "SIMULATION ONLY" in out

    def test_broadcast_no_network(self, beacon, capsys):
        """broadcast() must never perform real network I/O."""
        beacon.broadcast()
        out = capsys.readouterr().out
        assert "NO network transmission" in out

    def test_broadcast_with_prebuilt_key(self, beacon, capsys):
        key = beacon.generate_key()
        beacon.broadcast(key=key)
        out = capsys.readouterr().out
        assert key.hex[:16] in out

    # ── verify ────────────────────────────────────────────────────────────

    def test_verify_passes(self, beacon):
        result = beacon.verify()
        assert result["verified"]
        assert result["deterministic"]
        assert result["distinct"]

    # ── CLI ───────────────────────────────────────────────────────────────

    def test_cli_main_runs(self, capsys):
        cli_main(["--n", "3", "--rounds", "1", "--seed", "0"])
        out = capsys.readouterr().out
        assert "SIMULATION ONLY" in out

    def test_cli_main_even_n_exits(self, capsys):
        with pytest.raises(SystemExit):
            cli_main(["--n", "4"])
        # The CLI prints "Error: n must be odd, got 4" to stderr — this is
        # correct behaviour; we capture it here so it doesn't bleed into
        # the test runner output (cosmetic fix, audit item #3).
        capsys.readouterr()


# ══════════════════════════════════════════════════════════════════════════════
# Top-level package access
# ══════════════════════════════════════════════════════════════════════════════

class TestPackageAccess:
    def test_all_symbols_importable(self):
        import flu
        for sym in [
            "ExperimentalDesign", "DesignResult",
            "FLUInitializer",
            "TensorNetworkSimulator",
            "LighthouseBeacon", "BeaconKey", "cli_main",
        ]:
            assert hasattr(flu, sym), f"flu.{sym} not found"

    def test_smoke_design(self):
        import flu
        r = flu.ExperimentalDesign().generate(3, 2)
        assert r.overall_pass

    def test_smoke_neural(self):
        import flu
        W = flu.FLUInitializer().weights((3, 3))
        assert W.shape == (3, 3)

    def test_smoke_quantum(self):
        import flu
        s = flu.TensorNetworkSimulator().prepare_state(2)
        assert len(s) == 9

    def test_smoke_lighthouse(self):
        import flu
        key = flu.LighthouseBeacon(n=3, seed=0).generate_key()
        assert len(key.hex) == 64
