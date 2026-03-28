"""
flu.applications.bo
====================
Bayesian Optimisation with FLU-Owen quasi-random initialisation.

This package demonstrates the empirical gains predicted by the DN2 theorem family:

  DN2-ETK  → discrepancy constant C_APN(D) = C_classic·(B/√n)^D
  DN2-VAR  → integration variance (B/√n)^{2D} times smaller than standard Owen
  DN2-ANOVA→ high-order interactions suppressed by (B/√n)^{2|u|}

The BO loop uses a minimal pure-numpy GP (RBF kernel + EI acquisition) so
that performance differences are attributable to the initial design quality,
not to the surrogate or acquisition machinery.

Quick start
-----------
    from flu.applications.bo import run_benchmark
    run_benchmark()              # prints table and saves plots

    from flu.applications.bo.optimizer import run_bo
    from flu.applications.bo.benchmarks import hartmann6
    result = run_bo(hartmann6, D=6, global_min=-3.322, sampler="flu_owen")
    print(f"Best found: {result.final_best:.4f}  gap closed: {result.gap_closed:.1%}")
"""
from .benchmarks import BENCHMARKS, branin, hartmann3, hartmann6, ackley, rosenbrock
from .samplers   import SAMPLERS, flu_owen, sobol, halton, latin_hypercube, random_mc
from .gp         import GaussianProcess, expected_improvement, optimize_hyperparams
from .optimizer  import BOResult, BenchmarkSummary, run_bo, run_bo_multi

def run_benchmark(
    functions:   list[str] | None = None,
    samplers:    list[str] | None = None,
    n_seeds:     int = 8,
    N_init:      int = 20,
    N_iter:      int = 40,
    save_plots:  bool = True,
    output_dir:  str = ".",
    verbose:     bool = True,
) -> dict:
    """
    Run the full BO benchmark and return summary dict.
    See examples/bo_benchmark.py for a standalone script.
    """
    from .runner import run_full_benchmark
    return run_full_benchmark(
        functions=functions, samplers=samplers,
        n_seeds=n_seeds, N_init=N_init, N_iter=N_iter,
        save_plots=save_plots, output_dir=output_dir, verbose=verbose,
    )

__all__ = [
    "BENCHMARKS", "SAMPLERS",
    "branin", "hartmann3", "hartmann6", "ackley", "rosenbrock",
    "flu_owen", "sobol", "halton", "latin_hypercube", "random_mc",
    "GaussianProcess", "expected_improvement", "optimize_hyperparams",
    "BOResult", "BenchmarkSummary", "run_bo", "run_bo_multi",
    "run_benchmark",
]
