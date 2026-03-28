"""
flu.applications.bo.samplers
==============================
Initial-design samplers for BO benchmarking.

All samplers produce N points in [0, 1]^D.

Samplers
--------
flu_owen(n_base, D, N, seed_rank)   FLU-Owen scrambled FractalNetKinetic (DN2 PROVEN)
sobol(D, N, seed)                   Scrambled Sobol' (scipy.stats.qmc)
halton(D, N)                        Halton sequence (scipy.stats.qmc)
latin_hypercube(D, N, seed)         Optimised LHC (scipy.stats.qmc)
random_mc(D, N, seed)               Pure Monte Carlo uniform random

Theorem connections
-------------------
FLU-Owen: DN2-ETK  → C_APN(D) = C_classic(D)·(B/√n)^D  (discrepancy constant)
          DN2-VAR  → Var[I_N] ≤ C·(B/√n)^{2D}·(log N)^{D-1}/N^3  (variance)
          DN2-ANOVA→ subset u suppressed by (B/√n)^{2|u|}   (effective dimension)
"""
from __future__ import annotations
import math
import numpy as np

# ── FLU-Owen ──────────────────────────────────────────────────────────────────

def flu_owen(D: int, N: int,
             n_base: int = 5,
             seed_rank: int = 0) -> np.ndarray:
    """
    FLU-Owen scrambled FractalNetKinetic points in [0,1)^D.

    n_base: radix — must be an odd prime with APN seeds in GOLDEN_SEEDS.
            n=5 gives B=1.000 (Weil tight), improvement (√5)^D per depth.
            n=7 gives B=1.152, improvement (√7/1.152)^D ≈ 1.58× vs Sobol'.
    N:      number of points requested (rounded up to next power of n_base^D).
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    from flu.core.fractal_net import FractalNetKinetic

    net = FractalNetKinetic(n=n_base, d=D)
    pts = net.generate_scrambled(N, seed_rank=seed_rank, mode="owen")
    return pts[:N]


def flu_owen_multi(D: int, N: int,
                   n_base: int = 5,
                   n_seeds: int = 4) -> list[np.ndarray]:
    """
    Return n_seeds independent FLU-Owen samples (different seed_ranks).
    Useful for variance estimation across runs.
    """
    return [flu_owen(D, N, n_base=n_base, seed_rank=k) for k in range(n_seeds)]


# ── Sobol' ────────────────────────────────────────────────────────────────────

def sobol(D: int, N: int, seed: int = 42) -> np.ndarray:
    """Scrambled Sobol' sequence via scipy.stats.qmc."""
    from scipy.stats.qmc import Sobol as SobolQMC
    # Sobol requires N = power of 2; round up
    m = math.ceil(math.log2(max(N, 1)))
    sampler = SobolQMC(d=D, scramble=True, seed=seed)
    pts = sampler.random(2**m)
    return pts[:N]


def sobol_multi(D: int, N: int, n_seeds: int = 4) -> list[np.ndarray]:
    return [sobol(D, N, seed=s) for s in range(n_seeds)]


# ── Halton ────────────────────────────────────────────────────────────────────

def halton(D: int, N: int) -> np.ndarray:
    """Halton sequence via scipy.stats.qmc."""
    from scipy.stats.qmc import Halton as HaltonQMC
    sampler = HaltonQMC(d=D, scramble=True)
    return sampler.random(N)


# ── Latin Hypercube ───────────────────────────────────────────────────────────

def latin_hypercube(D: int, N: int, seed: int = 42) -> np.ndarray:
    """Optimised Latin Hypercube Sample via scipy.stats.qmc."""
    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=D, seed=seed)
    return sampler.random(N)


def latin_hypercube_multi(D: int, N: int, n_seeds: int = 4) -> list[np.ndarray]:
    return [latin_hypercube(D, N, seed=s) for s in range(n_seeds)]


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def random_mc(D: int, N: int, seed: int = 42) -> np.ndarray:
    """Pure uniform random (Monte Carlo baseline)."""
    rng = np.random.default_rng(seed)
    return rng.random((N, D))


def random_mc_multi(D: int, N: int, n_seeds: int = 4) -> list[np.ndarray]:
    return [random_mc(D, N, seed=s) for s in range(n_seeds)]


# ── Sampler registry ──────────────────────────────────────────────────────────

SAMPLERS: dict = {
    "flu_owen":  {"fn": flu_owen,       "multi": flu_owen_multi,         "label": "FLU-Owen (n=5)",   "color": "#E05C2E"},
    "flu_owen7": {"fn": lambda D,N: flu_owen(D,N,n_base=7),
                  "multi": lambda D,N,k: flu_owen_multi(D,N,n_base=7,n_seeds=k),
                  "label": "FLU-Owen (n=7)",   "color": "#C03020"},
    "sobol":     {"fn": sobol,          "multi": sobol_multi,            "label": "Sobol' (scrambled)","color": "#2E6CB8"},
    "halton":    {"fn": halton,         "multi": lambda D,N,k: [halton(D,N)]*k,
                  "label": "Halton",           "color": "#5B9E3F"},
    "lhc":       {"fn": latin_hypercube,"multi": latin_hypercube_multi,  "label": "Latin HyperCube",  "color": "#9B59B6"},
    "mc":        {"fn": random_mc,      "multi": random_mc_multi,        "label": "Monte Carlo",      "color": "#888888"},
}
