"""
flu.applications.bo.benchmarks
================================
Standard BO benchmark functions.

All functions operate on x ∈ [0, 1]^D and return a scalar.
Convention: minimisation target (lower is better).
Negative sign applied where needed to turn maximisation into minimisation.

Functions
---------
branin(x)       D=2    global min ≈ 0.397
hartmann3(x)    D=3    global min ≈ -3.863
hartmann6(x)    D=6    global min ≈ -3.322
ackley(x)       D=2–10 global min = 0  at x=0.5
rosenbrock(x)   D=2–10 global min = 0  at x=0.4,0.4,...
"""
from __future__ import annotations
import math
import numpy as np


# ── Branin ────────────────────────────────────────────────────────────────────

def branin(x: np.ndarray) -> float:
    """Branin function on [0,1]^2. Global min ≈ 0.397 at 3 locations."""
    assert x.shape == (2,), f"Branin requires D=2, got {x.shape}"
    # Remap [0,1]^2 → x1 ∈ [-5,10], x2 ∈ [0,15]
    x1 = 15 * x[0] - 5
    x2 = 15 * x[1]
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*math.cos(x1) + s)


# ── Hartmann ──────────────────────────────────────────────────────────────────

_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
_H3_P = 1e-4 * np.array([[3689, 1170, 2673],
                          [4699, 4387, 7470],
                          [1091, 8732, 5547],
                          [ 381, 5743, 8828]])

def hartmann3(x: np.ndarray) -> float:
    """Hartmann-3D on [0,1]^3. Global min ≈ -3.863."""
    assert x.shape == (3,), f"Hartmann3 requires D=3, got {x.shape}"
    inner = np.sum(_H3_A * (x - _H3_P)**2, axis=1)
    return float(-np.dot(_H3_ALPHA, np.exp(-inner)))


_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array([
    [10,  3,  17, 3.5, 1.7,  8],
    [0.05,10,  17, 0.1,  8, 14],
    [ 3, 3.5,  1.7, 10,  17,  8],
    [17,  8, 0.05, 10, 0.1, 14],
])
_H6_P = 1e-4 * np.array([
    [1312, 1696, 5569,  124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091,  381],
])

def hartmann6(x: np.ndarray) -> float:
    """Hartmann-6D on [0,1]^6. Global min ≈ -3.322."""
    assert x.shape == (6,), f"Hartmann6 requires D=6, got {x.shape}"
    inner = np.sum(_H6_A * (x - _H6_P)**2, axis=1)
    return float(-np.dot(_H6_ALPHA, np.exp(-inner)))


# ── Ackley ────────────────────────────────────────────────────────────────────

def ackley(x: np.ndarray) -> float:
    """Ackley on [0,1]^D, centred at 0.5. Global min = 0 at x = 0.5."""
    z = 2 * x - 1.0                       # remap to [-1, 1]^D
    D = len(z)
    a, b, c = 20.0, 0.2, 2 * math.pi
    sum_sq   = np.sum(z**2)
    sum_cos  = np.sum(np.cos(c * z))
    return float(-a * math.exp(-b * math.sqrt(sum_sq / D))
                 - math.exp(sum_cos / D) + a + math.e)


# ── Rosenbrock ────────────────────────────────────────────────────────────────

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock on [0,1]^D, optimal at x ≈ 0.4,0.4,...  Min = 0."""
    z = 4 * x - 1.5                        # remap roughly to [-1.5, 2.5]
    val = sum(100*(z[i+1] - z[i]**2)**2 + (1 - z[i])**2
              for i in range(len(z) - 1))
    return float(val)


# ── Registry ──────────────────────────────────────────────────────────────────

BENCHMARKS: dict = {
    "branin":     {"fn": branin,     "dim": 2,  "global_min": 0.397,  "desc": "Branin (D=2)"},
    "hartmann3":  {"fn": hartmann3,  "dim": 3,  "global_min": -3.863, "desc": "Hartmann-3 (D=3)"},
    "hartmann6":  {"fn": hartmann6,  "dim": 6,  "global_min": -3.322, "desc": "Hartmann-6 (D=6)"},
    "ackley5":    {"fn": ackley,     "dim": 5,  "global_min": 0.0,    "desc": "Ackley (D=5)"},
    "rosenbrock4":{"fn": rosenbrock, "dim": 4,  "global_min": 0.0,    "desc": "Rosenbrock (D=4)"},
}
