#!/usr/bin/env python3
"""
examples/bo_benchmark.py
=========================
Standalone Bayesian Optimisation benchmark comparing FLU-Owen initial designs
against Sobol', Latin Hypercube, and Monte Carlo baselines.

Usage
-----
    python examples/bo_benchmark.py                   # full run
    python examples/bo_benchmark.py --quick           # fast 3-function run
    python examples/bo_benchmark.py --function hartmann6 --seeds 12

Background
----------
The FLU-Owen sampler uses FractalNetKinetic.generate_scrambled(mode="owen"),
proven in V15.3 to achieve:

  - Discrepancy constant: C_APN(D) = C_classic(D)·(B/√n)^D
    e.g. n=5, D=3 → 11.2× better than unscrambled
  - Variance: (B/√n)^{2D} improvement, independent of function smoothness
  - ANOVA: high-order interactions suppressed by (B/√n)^{2|u|}
    effective integration dimension approximately halved

The BO setting is a natural test because:
1. Initial design quality directly controls the first GP fit.
2. Better discrepancy → less correlation in the initial design → more
   informative GP posterior → fewer BO iterations to converge.
3. The effective dimension reduction predicts the advantage is larger
   for higher-D functions (Hartmann-6 > Branin, consistent with theory).

References
----------
  DN2-ETK, DN2-VAR, DN2-ANOVA: docs/PROOF_DN2_APN_SCRAMBLING.md
  Owen (1997): Monte Carlo variance of scrambled net quadrature
"""
from __future__ import annotations
import sys, os, argparse, time

# Make flu importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flu.applications.bo import run_benchmark
from flu.applications.bo.benchmarks import BENCHMARKS


def parse_args():
    p = argparse.ArgumentParser(description="FLU-Owen BO benchmark")
    p.add_argument("--quick",    action="store_true",
                   help="Quick run: branin/hartmann3/hartmann6, 4 seeds, 15+20 iters")
    p.add_argument("--function", default=None,
                   choices=list(BENCHMARKS.keys()),
                   help="Run a single function only")
    p.add_argument("--seeds",    type=int, default=8,
                   help="Number of random seeds per (sampler, function) pair")
    p.add_argument("--n_init",   type=int, default=20,
                   help="Initial design size")
    p.add_argument("--n_iter",   type=int, default=40,
                   help="BO iterations after initial design")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip matplotlib output")
    p.add_argument("--outdir",   default=".",
                   help="Directory for plot files")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        functions = ["branin", "hartmann3", "hartmann6"]
        n_seeds, N_init, N_iter = 4, 15, 20
        print("Quick mode: 3 functions × 4 seeds × 35 evals each")
    elif args.function:
        functions = [args.function]
        n_seeds, N_init, N_iter = args.seeds, args.n_init, args.n_iter
    else:
        functions = list(BENCHMARKS.keys())
        n_seeds, N_init, N_iter = args.seeds, args.n_init, args.n_iter

    samplers = ["flu_owen", "sobol", "lhc", "mc"]

    print("=" * 64)
    print("  FLU-Owen Bayesian Optimisation Benchmark")
    print("  Theorem basis: DN2-ETK, DN2-VAR, DN2-ANOVA (V15.3)")
    print(f"  Functions: {functions}")
    print(f"  Samplers:  {samplers}")
    print(f"  Seeds:     {n_seeds}   N_init: {N_init}   N_iter: {N_iter}")
    print("=" * 64)

    t0 = time.perf_counter()
    results = run_benchmark(
        functions   = functions,
        samplers    = samplers,
        n_seeds     = n_seeds,
        N_init      = N_init,
        N_iter      = N_iter,
        save_plots  = not args.no_plots,
        output_dir  = args.outdir,
        verbose     = True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # ── Per-function winner ────────────────────────────────────────────────────
    print("\n  Winners by gap closed:")
    for fn_name in functions:
        winner = max(results[fn_name].items(),
                     key=lambda kv: kv[1].gap_closed_mean)
        print(f"    {BENCHMARKS[fn_name]['desc']:22s}: {winner[0]:12s}"
              f"  {winner[1].gap_closed_mean:.1%}")


if __name__ == "__main__":
    main()
