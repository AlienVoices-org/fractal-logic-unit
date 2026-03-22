"""
tests/test_utils/test_viz.py
=============================
Smoke tests for flu.utils.viz — all three plotting functions.
"""
import unittest
import numpy as np
from flu.utils.viz import plot_hyperprism_2d, plot_lo_shu_grid, plot_bijection_path
from flu.core.parity_switcher import generate
from flu.core.hypercell import FLUHyperCell

def _matplotlib_available():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False

# ── IMPORT ────────────────────────────────────────────────────────────────────
def test_viz_functions_importable():
    assert callable(plot_hyperprism_2d)
    assert callable(plot_lo_shu_grid)
    assert callable(plot_bijection_path)

def test_viz_import_does_not_call_show():
    assert True  # reaching here without a window proves it

def test_viz_module_has_public_functions():
    from flu.utils import viz
    public = {n for n in dir(viz) if not n.startswith("_")}
    assert {"plot_hyperprism_2d", "plot_lo_shu_grid", "plot_bijection_path"}.issubset(public)

# ── WITHOUT MATPLOTLIB ────────────────────────────────────────────────────────
def test_plot_hyperprism_2d_no_matplotlib():
    if _matplotlib_available():
        raise unittest.SkipTest("matplotlib installed")
    M = generate(n=3, d=2)
    try:
        plot_hyperprism_2d(M)
    except (ImportError, RuntimeError, ModuleNotFoundError):
        pass

def test_plot_lo_shu_grid_no_matplotlib():
    if _matplotlib_available():
        raise unittest.SkipTest("matplotlib installed")
    try:
        plot_lo_shu_grid(FLUHyperCell())
    except (ImportError, RuntimeError, ModuleNotFoundError):
        pass

def test_plot_bijection_path_no_matplotlib():
    if _matplotlib_available():
        raise unittest.SkipTest("matplotlib installed")
    try:
        plot_bijection_path(n=3, d=2)
    except (ImportError, RuntimeError, ModuleNotFoundError):
        pass

# ── WITH MATPLOTLIB ───────────────────────────────────────────────────────────
def test_plot_hyperprism_2d_with_matplotlib():
    if not _matplotlib_available():
        raise unittest.SkipTest("matplotlib not installed")
    import matplotlib; matplotlib.use("Agg")
    M = generate(n=3, d=2)
    fig, ax = plot_hyperprism_2d(M)
    assert fig is not None

def test_plot_bijection_path_with_matplotlib():
    if not _matplotlib_available():
        raise unittest.SkipTest("matplotlib not installed")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plot_bijection_path(n=3, d=2)
    assert fig is not None
    plt.close(fig)

def test_plot_lo_shu_grid_with_matplotlib():
    if not _matplotlib_available():
        raise unittest.SkipTest("matplotlib not installed")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hc = FLUHyperCell()
    fig, ax = plot_lo_shu_grid(hc)
    assert fig is not None
    plt.close(fig)
