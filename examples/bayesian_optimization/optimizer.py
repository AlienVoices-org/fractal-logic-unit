"""
flu.applications.bo.optimizer
================================
Bayesian Optimisation loop and benchmark runner.

BOResult      — holds the full trajectory of one BO run
run_bo        — run one BO experiment (one sampler, one function)
benchmark     — run a full comparison across samplers and functions
"""
from __future__ import annotations
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .gp import GaussianProcess, optimize_hyperparams, expected_improvement
from .samplers import flu_owen, sobol, halton, latin_hypercube, random_mc


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BOResult:
    """Full trajectory of one BO run."""
    sampler_name:   str
    function_name:  str
    D:              int
    N_init:         int
    N_total:        int
    global_min:     float

    # per-iteration
    f_values:       np.ndarray    # shape (N_total,)
    best_so_far:    np.ndarray    # shape (N_total,)  running minimum
    X_evaluated:    np.ndarray    # shape (N_total, D)
    wall_time_s:    float = 0.0

    @property
    def final_best(self) -> float:
        return float(self.best_so_far[-1])

    @property
    def gap_closed(self) -> float:
        """Fraction of gap to global min closed: 1 = perfect."""
        f0 = self.f_values[:self.N_init].min()
        g  = self.global_min
        improvement = f0 - self.final_best
        available   = f0 - g
        if abs(available) < 1e-10:
            return 1.0
        return float(np.clip(improvement / available, 0.0, 1.0))

    @property
    def regret(self) -> np.ndarray:
        """Simple regret: best_so_far - global_min."""
        return self.best_so_far - self.global_min

    @property
    def n_evals_to_threshold(self, threshold_pct: float = 0.90) -> int:
        """Number of evaluations to close threshold_pct of the gap."""
        f0  = self.f_values[:self.N_init].min()
        g   = self.global_min
        tgt = f0 - threshold_pct * (f0 - g)
        for i, b in enumerate(self.best_so_far):
            if b <= tgt:
                return i + 1
        return self.N_total


# ── BO loop ────────────────────────────────────────────────────────────────────

def run_bo(
    fn:              Callable[[np.ndarray], float],
    D:               int,
    global_min:      float,
    sampler:         str = "flu_owen",
    N_init:          int = 20,
    N_iter:          int = 30,
    n_base:          int = 5,
    seed_rank:       int = 0,
    acq_candidates:  int = 1000,
    optimize_hp:     bool = True,
    verbose:         bool = False,
) -> BOResult:
    """
    Run one Bayesian Optimisation experiment.

    Phase 1 — Initial design (N_init points) via the chosen sampler.
    Phase 2 — BO loop (N_iter iterations): GP fit → EI maximisation →
               evaluate → update.

    Acquisition maximisation: draw acq_candidates random candidates (uniform)
    and pick the one with highest EI. Simple but avoids gradient issues.
    """
    t0 = time.perf_counter()

    # ── Phase 1: initial design ───────────────────────────────────────────────
    if sampler == "flu_owen":
        X_init = flu_owen(D, N_init, n_base=n_base, seed_rank=seed_rank)
    elif sampler == "sobol":
        X_init = sobol(D, N_init, seed=seed_rank)
    elif sampler == "halton":
        X_init = halton(D, N_init)
    elif sampler == "lhc":
        X_init = latin_hypercube(D, N_init, seed=seed_rank)
    elif sampler == "mc":
        X_init = random_mc(D, N_init, seed=seed_rank)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    X_init = np.clip(X_init[:N_init], 0.0, 1.0)
    y_init = np.array([fn(x) for x in X_init])

    X_eval = list(X_init)
    y_eval = list(y_init)

    f_values    = list(y_init)
    best_so_far = [min(y_init[:i+1]) for i in range(N_init)]

    if verbose:
        print(f"  [{sampler:12s}] init best: {min(y_init):.4f}  (global min: {global_min:.4f})")

    # ── Phase 2: BO loop ──────────────────────────────────────────────────────
    for it in range(N_iter):
        X_arr = np.array(X_eval)
        y_arr = np.array(y_eval)

        # Normalise y for numerical stability
        y_mu, y_std = y_arr.mean(), max(y_arr.std(), 1e-6)
        y_norm = (y_arr - y_mu) / y_std

        # Fit GP
        if optimize_hp and len(y_arr) % 10 == 0:
            gp = optimize_hyperparams(X_arr, y_norm, n_restarts=2)
        else:
            gp = GaussianProcess(length_scale=0.3, sigma_f=1.0, sigma_n=1e-3)
            gp.fit(X_arr, y_norm)

        # EI candidates
        rng = np.random.default_rng(it + seed_rank * 1000)
        X_cand = rng.random((acq_candidates, D))
        mu_c, std_c = gp.predict(X_cand)
        # Un-normalise for f_best comparison
        mu_c   = mu_c * y_std + y_mu
        std_c  = std_c * y_std
        f_best = min(y_eval)
        ei     = expected_improvement(mu_c, std_c, f_best)
        x_next = X_cand[np.argmax(ei)]

        y_next = fn(x_next)
        X_eval.append(x_next)
        y_eval.append(y_next)
        f_values.append(y_next)
        best_so_far.append(min(best_so_far[-1], y_next))

        if verbose and (it + 1) % 10 == 0:
            print(f"  [{sampler:12s}] iter {it+1:3d}: best = {best_so_far[-1]:.4f}")

    wall_time = time.perf_counter() - t0

    return BOResult(
        sampler_name  = sampler,
        function_name = fn.__name__ if hasattr(fn, '__name__') else "unknown",
        D             = D,
        N_init        = N_init,
        N_total       = N_init + N_iter,
        global_min    = global_min,
        f_values      = np.array(f_values),
        best_so_far   = np.array(best_so_far),
        X_evaluated   = np.array(X_eval),
        wall_time_s   = wall_time,
    )


# ── Multi-seed runner ────────────────────────────────────────────────────────

def run_bo_multi(
    fn:           Callable,
    D:            int,
    global_min:   float,
    sampler:      str,
    n_seeds:      int = 8,
    N_init:       int = 20,
    N_iter:       int = 30,
    n_base:       int = 5,
    verbose:      bool = False,
) -> list[BOResult]:
    """Run BO n_seeds times with different seed_ranks for variance estimation."""
    results = []
    for seed in range(n_seeds):
        r = run_bo(fn, D, global_min,
                   sampler=sampler, N_init=N_init, N_iter=N_iter,
                   n_base=n_base, seed_rank=seed, verbose=verbose)
        results.append(r)
    return results


# ── Full benchmark ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkSummary:
    """Aggregated results across seeds for one (sampler, function) pair."""
    sampler_name:  str
    function_name: str
    D:             int
    N_init:        int
    N_total:       int
    global_min:    float
    n_seeds:       int

    # Aggregated metrics
    final_best_mean:   float
    final_best_std:    float
    gap_closed_mean:   float
    gap_closed_std:    float
    regret_curve_mean: np.ndarray   # shape (N_total,)
    regret_curve_std:  np.ndarray
    wall_time_mean:    float

    @classmethod
    def from_results(cls, results: list[BOResult]) -> "BenchmarkSummary":
        r0   = results[0]
        fb   = np.array([r.final_best    for r in results])
        gc   = np.array([r.gap_closed    for r in results])
        regs = np.stack([r.regret        for r in results])
        wt   = np.array([r.wall_time_s   for r in results])
        return cls(
            sampler_name     = r0.sampler_name,
            function_name    = r0.function_name,
            D                = r0.D,
            N_init           = r0.N_init,
            N_total          = r0.N_total,
            global_min       = r0.global_min,
            n_seeds          = len(results),
            final_best_mean  = float(fb.mean()),
            final_best_std   = float(fb.std()),
            gap_closed_mean  = float(gc.mean()),
            gap_closed_std   = float(gc.std()),
            regret_curve_mean= regs.mean(axis=0),
            regret_curve_std = regs.std(axis=0),
            wall_time_mean   = float(wt.mean()),
        )

    def __str__(self) -> str:
        return (
            f"[{self.sampler_name:12s} | {self.function_name:12s} D={self.D}] "
            f"best={self.final_best_mean:.4f}±{self.final_best_std:.4f}  "
            f"gap_closed={self.gap_closed_mean:.1%}±{self.gap_closed_std:.1%}  "
            f"({self.n_seeds} seeds, {self.wall_time_mean:.1f}s/run)"
        )
