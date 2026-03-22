"""
examples/quickstart.py
======================
FLUHyperCell in 20 lines — the fastest path to understanding FLU.

Run:  PYTHONPATH=src python3 examples/quickstart.py
"""

import flu

# ── Build a 3⁴ HyperCell ──────────────────────────────────────────────────────
hc     = flu.FLUHyperCell()
result = hc.verify()

print("=== FLUHyperCell quickstart ===\n")
print(f"Fully verified:       {result['fully_verified']}")
print(f"Seam verified:        {result['seam_verified']}")

# ── Centre cell ───────────────────────────────────────────────────────────────
centre = hc.center()
print(f"\nCentre cell:")
print(f"  norm0    = {centre.norm0}   (FM-Dance step index)")
print(f"  norm1    = {centre.norm1}   (1-based index)")
print(f"  balanced = {centre.balanced}")
print(f"  bt       = {centre.bt}  (balanced-ternary 4-tuple)")

# ── Sparse manifold address ───────────────────────────────────────────────────
print(f"\n4D sparse addresses:")
for r, c in [(0, 0), (4, 4), (8, 8)]:
    coords = hc.sparse_address(r, c)
    print(f"  cell({r},{c}) → {coords}")

# ── Pivot-constrained slice ───────────────────────────────────────────────────
centre_plane = hc.cells_with_pivot(pivot_value=0, dimension=0)
print(f"\nCells with coords[0]=0: {len(centre_plane)} cells (expected 27)")

# ── 3⁶ Fractal embedding ──────────────────────────────────────────────────────
frac   = hc.embed_as_3_6()
verify = frac.verify(silent=True)
print(f"\n3⁶ FractalHyperCell:")
print(f"  Total cells:       {len(frac)}")
print(f"  Unique 6D addrs:   {verify['unique_addresses']}")
print(f"  Seam verified:     {verify['seam_verified']}")

# ── Factoradic ↔ FM-Dance bridge ──────────────────────────────────────────────
print(f"\nFactoradic ↔ FM-Dance bridge (ITER-3A):")
for k in range(3):
    step = flu.factoradic_to_fm_coords(k=k, n=3, d=4, pivot_dim=0, pivot_val=0)
    print(f"  k={k}  fm_coords={step.fm_coords}  arrow={step.arrow.tolist()}")

print("\nDone.")
