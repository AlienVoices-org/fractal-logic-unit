"""
examples/experimental_design.py
================================
ExperimentalDesign — 5-factor Latin Hypercube DOE.

Demonstrates:
  - Odd n_levels  (FM-Dance path, STATUS: PROVEN)
  - Even n_levels (sum-mod path,  STATUS: PROVEN)
  - Stratified sampling
  - Optional pandas export

Run:  PYTHONPATH=src python3 examples/experimental_design.py
"""

import numpy as np
from flu import ExperimentalDesign

ed = ExperimentalDesign()

print("=== Latin Hypercube Experimental Design ===\n")

# ── 5-factor DOE (odd n=5, d=3) ───────────────────────────────────────────────
result = ed.generate(
    n_levels    = 5,
    n_factors   = 3,
    factor_names = ["Temperature", "Pressure", "FlowRate"],
)

print(f"Design: n_levels={result.n_levels}, n_factors={result.n_factors}")
print(f"Matrix shape:   {result.matrix.shape}")
print(f"Overall pass:   {result.overall_pass}")
print(f"Latin OK:       {result.report['latin']['latin_ok']}")
print(f"Coverage OK:    {result.report['coverage']['coverage_ok']}")
print(f"\nMatrix (first 2D slice at index 0):\n{result.matrix[0]}")

# ── Stratified sample ─────────────────────────────────────────────────────────
sample = ed.stratified_sample(result, n_samples=3, rng=np.random.default_rng(42))
print(f"\nStratified sample (3 runs × 3 factors):\n{sample}")
print(f"Columns: {result.factor_names}")

# ── Even n_levels (n=6) ───────────────────────────────────────────────────────
result_even = ed.generate(n_levels=6, n_factors=2,
                           factor_names=["Voltage", "Current"])
print(f"\nEven n=6, d=2 overall_pass: {result_even.overall_pass}")

# ── Optional pandas export ────────────────────────────────────────────────────
try:
    df = ed.to_dataframe(result)
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())
except ImportError:
    print("\n(pandas not installed — skipping DataFrame export)")

print("\nDone.")
