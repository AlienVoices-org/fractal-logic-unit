"""
flu.applications.bo.runner
============================
Full benchmark runner: aggregates multi-seed BO results, prints a summary
table, and produces comparison plots (matplotlib, optional).
"""
from __future__ import annotations
import os
import sys
import numpy as np
from typing import Optional

from .benchmarks import BENCHMARKS
from .optimizer  import run_bo_multi, BenchmarkSummary


# ── Default sampler list ──────────────────────────────────────────────────────

DEFAULT_SAMPLERS = ["flu_owen", "sobol", "lhc", "mc"]


# ── Main runner ───────────────────────────────────────────────────────────────

def run_full_benchmark(
    functions:   Optional[list[str]] = None,
    samplers:    Optional[list[str]] = None,
    n_seeds:     int  = 8,
    N_init:      int  = 20,
    N_iter:      int  = 40,
    save_plots:  bool = True,
    output_dir:  str  = ".",
    verbose:     bool = True,
) -> dict:
    """
    Run BO benchmark for all (function, sampler) combinations.

    Returns nested dict: results[fn_name][sampler_name] = BenchmarkSummary
    """
    if functions is None:
        functions = list(BENCHMARKS.keys())
    if samplers is None:
        samplers = DEFAULT_SAMPLERS

    results = {}
    total_runs = len(functions) * len(samplers)
    run_idx    = 0

    for fn_name in functions:
        spec = BENCHMARKS[fn_name]
        fn, D, global_min, desc = spec["fn"], spec["dim"], spec["global_min"], spec["desc"]
        results[fn_name] = {}

        if verbose:
            print(f"\n{'─'*62}")
            print(f"  {desc}   global_min={global_min:.4f}   {n_seeds} seeds each")
            print(f"{'─'*62}")

        for sampler_name in samplers:
            run_idx += 1
            if verbose:
                print(f"  [{run_idx}/{total_runs}] {sampler_name:12s} × {fn_name} ...", end="", flush=True)

            # Tag function name for BOResult
            def tagged_fn(x, _fn=fn, _name=fn_name):
                v = _fn(x)
                tagged_fn.__name__ = _name
                return v
            tagged_fn.__name__ = fn_name

            seed_results = run_bo_multi(
                fn=tagged_fn, D=D, global_min=global_min,
                sampler=sampler_name, n_seeds=n_seeds,
                N_init=N_init, N_iter=N_iter,
                verbose=False,
            )
            summary = BenchmarkSummary.from_results(seed_results)
            results[fn_name][sampler_name] = summary

            if verbose:
                print(f" best={summary.final_best_mean:.4f}±{summary.final_best_std:.3f}"
                      f"  gap={summary.gap_closed_mean:.0%}")

    # ── Print summary table ───────────────────────────────────────────────────
    if verbose:
        _print_table(results, functions, samplers)

    # ── Save plots ────────────────────────────────────────────────────────────
    if save_plots:
        try:
            _save_plots(results, functions, samplers, output_dir, N_init)
            if verbose:
                print(f"\nPlots saved to: {os.path.abspath(output_dir)}")
        except ImportError:
            if verbose:
                print("\n(matplotlib not available — plots skipped)")

    return results


# ── Table printer ─────────────────────────────────────────────────────────────

def _print_table(results: dict, functions: list[str], samplers: list[str]) -> None:
    sampler_labels = {
        "flu_owen":  "FLU-Owen n=5",
        "flu_owen7": "FLU-Owen n=7",
        "sobol":     "Sobol'",
        "halton":    "Halton",
        "lhc":       "LHC",
        "mc":        "Monte Carlo",
    }
    col_w = 20

    print(f"\n{'═'*72}")
    print("  BO Benchmark Summary — gap closed (mean ± std) across seeds")
    print(f"{'═'*72}")

    # Header
    header = f"  {'Function':20s}"
    for s in samplers:
        header += f"  {sampler_labels.get(s, s):^{col_w}}"
    print(header)
    print(f"  {'-'*20}" + f"  {'-'*col_w}" * len(samplers))

    # Rows
    for fn_name in functions:
        desc = BENCHMARKS[fn_name]["desc"]
        row  = f"  {desc:20s}"
        best_gap = max(results[fn_name][s].gap_closed_mean for s in samplers)
        for s in samplers:
            sm = results[fn_name][s]
            cell = f"{sm.gap_closed_mean:.0%}±{sm.gap_closed_std:.0%}"
            marker = " ◀" if sm.gap_closed_mean >= best_gap - 0.01 else "  "
            row += f"  {cell:^{col_w}}{marker}"
        print(row)

    print(f"{'═'*72}")
    print("  ◀ = best (or within 1%) for that function\n")

    # Variance ratio (FLU-Owen vs Sobol')
    if "flu_owen" in samplers and "sobol" in samplers:
        print("  Variance ratio (FLU-Owen final_best_std vs Sobol'):")
        for fn_name in functions:
            flu_s = results[fn_name]["flu_owen"].final_best_std
            sob_s = results[fn_name]["sobol"].final_best_std
            ratio = sob_s / max(flu_s, 1e-9)
            print(f"    {BENCHMARKS[fn_name]['desc']:22s}: {ratio:.2f}× lower std")
        print()


# ── Plot generation ───────────────────────────────────────────────────────────

_COLORS = {
    "flu_owen":  "#E05C2E",
    "flu_owen7": "#B03010",
    "sobol":     "#2E6CB8",
    "halton":    "#5B9E3F",
    "lhc":       "#9B59B6",
    "mc":        "#888888",
}
_LABELS = {
    "flu_owen":  "FLU-Owen (n=5)",
    "flu_owen7": "FLU-Owen (n=7)",
    "sobol":     "Sobol'",
    "halton":    "Halton",
    "lhc":       "LHC",
    "mc":        "Monte Carlo",
}


def _save_plots(results: dict, functions: list[str], samplers: list[str],
                output_dir: str, N_init: int) -> None:
    import math as _math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(output_dir, exist_ok=True)
    n_fn   = len(functions)
    n_cols = min(n_fn, 3)
    n_rows = _math.ceil(n_fn / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6*n_cols, 4.5*n_rows),
                             squeeze=False)
    fig.suptitle("FLU-Owen vs QMC Baselines — Bayesian Optimisation\n"
                 "(Regret = best_so_far − global_min, lower is better)",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, fn_name in enumerate(functions):
        ax  = axes[idx // n_cols][idx % n_cols]
        fn_results = results[fn_name]
        spec = BENCHMARKS[fn_name]
        x_axis = np.arange(1, spec["dim"]*N_init + 2)   # rough x

        for s in samplers:
            sm    = fn_results[s]
            curve = sm.regret_curve_mean
            err   = sm.regret_curve_std
            xs    = np.arange(1, len(curve) + 1)
            c     = _COLORS.get(s, "#444")
            ax.semilogy(xs, np.maximum(curve, 1e-6),
                        color=c, linewidth=2,
                        label=_LABELS.get(s, s), zorder=3)
            ax.fill_between(xs,
                            np.maximum(curve - err, 1e-6),
                            curve + err,
                            color=c, alpha=0.12, zorder=2)

        ax.axvline(N_init, color="#aaa", linestyle="--", linewidth=1,
                   label=f"BO start (n={N_init})")
        ax.set_title(spec["desc"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Regret (log scale)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(curve))

    # Hide unused axes
    for idx in range(len(functions), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bo_benchmark_regret.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Bar chart: gap closed ──────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(7, 2*len(functions)), 5))
    x     = np.arange(len(functions))
    width = 0.8 / len(samplers)

    for i, s in enumerate(samplers):
        means = [results[fn][s].gap_closed_mean   for fn in functions]
        stds  = [results[fn][s].gap_closed_std    for fn in functions]
        bars  = ax2.bar(x + i*width - 0.4 + width/2, means, width,
                        yerr=stds, label=_LABELS.get(s, s),
                        color=_COLORS.get(s, "#444"), capsize=4, alpha=0.85)

    ax2.set_title("Gap Closed (fraction of f₀−fₒₚₜ eliminated)\nFLU-Owen vs QMC Baselines",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Gap closed (1.0 = optimal)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([BENCHMARKS[fn]["desc"] for fn in functions],
                         rotation=15, ha="right")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(1.0, color="#aaa", linestyle="--", linewidth=1)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "bo_benchmark_gap_closed.png")
    fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
