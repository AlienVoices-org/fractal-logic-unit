"""
examples/neural_init.py
=======================
FLUInitializer — bias-free weight initialisation for a small network.

Demonstrates unit-variance, ~zero-mean FLU weights as a structured
alternative to Xavier / He initialisation.

Run:  PYTHONPATH=src python3 examples/neural_init.py
"""

import numpy as np
from flu import FLUInitializer

init = FLUInitializer(signed=True)

print("=== FLU Bias-Free Weight Initialisation ===\n")

# ── Simple two-layer network shapes ───────────────────────────────────────────
shapes = [
    ("input → hidden  (3×3×3)", (3, 3, 3)),
    ("hidden → output (5×5)",   (5, 5)),
    ("conv kernel     (3×3×3×3)", (3, 3, 3, 3)),
]

for label, shape in shapes:
    W = init.weights(shape)
    print(f"Layer: {label}")
    print(f"  shape = {W.shape}")
    print(f"  mean  = {W.mean():.6f}  (target ≈ 0)")
    print(f"  std   = {W.std():.6f}   (target = 1.0)")
    print(f"  min   = {W.min():.4f},  max = {W.max():.4f}")
    print()

# ── bias_free_check ───────────────────────────────────────────────────────────
W = init.weights((3, 3, 3))
# Center the weights (already ~zero; check the assertion)
W_centered = W - W.mean()
try:
    init.bias_free_check(W_centered, atol=1e-6)
    print("bias_free_check passed on manually centred weights.")
except AssertionError as e:
    print(f"bias_free_check: {e}")

# ── Optional PyTorch wrapper ──────────────────────────────────────────────────
try:
    import torch
    param = init.to_torch_parameter(W)
    print(f"\nPyTorch parameter: {param.shape}, requires_grad={param.requires_grad}")
except ImportError:
    print("\n(PyTorch not installed — skipping torch wrapper demo)")

# ── Optional JAX wrapper ──────────────────────────────────────────────────────
try:
    import jax
    arr = init.to_jax_array(W)
    print(f"JAX array: {arr.shape}, dtype={arr.dtype}")
except ImportError:
    print("(JAX not installed — skipping JAX wrapper demo)")

print("\nDone.")
