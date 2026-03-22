"""
flu/utils/viz.py
================
Optional visualisation utilities for FLU hyperprisms and Lo Shu grids.

All functions are guarded behind ``try: import matplotlib``.
If matplotlib is not installed, importing this module raises ImportError
with an actionable install hint.

Install with:  pip install flu[viz]   (or: pip install matplotlib)

STATUS: DESIGN INTENT — visual helpers, no mathematical claims.

Dependencies: matplotlib (optional), numpy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _require_matplotlib() -> None:
    """Raise ImportError with install hint if matplotlib is unavailable."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for flu.utils.viz.\n"
            "Install with:  pip install flu[viz]   (or: pip install matplotlib)"
        )


# ── plot_hyperprism_2d ────────────────────────────────────────────────────────

def plot_hyperprism_2d(
    array   : np.ndarray,
    title   : str               = "FLU Hyperprism (2D slice)",
    ax      : Optional["Axes"]  = None,
    cmap    : str               = "RdBu_r",
    annotate: bool              = True,
) -> Tuple["Figure", "Axes"]:
    """
    Heatmap of a 2D slice of a FLU hyperprism.

    Parameters
    ----------
    array    : np.ndarray  2D array (any dtype)
    title    : str         plot title
    ax       : Axes | None  existing axes to draw into; creates new figure if None
    cmap     : str         matplotlib colormap (default 'RdBu_r', diverging)
    annotate : bool        overlay integer values on each cell (default True)

    Returns
    -------
    (Figure, Axes)

    Raises
    ------
    ImportError  if matplotlib is not installed
    ValueError   if array is not 2D
    """
    _require_matplotlib()

    if array.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {array.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, array.shape[1] * 0.6),
                                        max(4, array.shape[0] * 0.6)))
    else:
        fig = ax.get_figure()

    vabs = max(abs(int(array.min())), abs(int(array.max())), 1)
    im   = ax.imshow(array, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="equal")

    if annotate:
        rows, cols = array.shape
        fontsize   = max(5, min(12, int(120 / max(rows, cols))))
        for r in range(rows):
            for c in range(cols):
                val = int(array[r, c])
                # White text on dark cells, dark on light
                brightness = (val + vabs) / (2 * vabs) if vabs > 0 else 0.5
                color      = "white" if brightness < 0.45 or brightness > 0.75 else "black"
                ax.text(c, r, str(val), ha="center", va="center",
                        fontsize=fontsize, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_xticks(range(array.shape[1]))
    ax.set_yticks(range(array.shape[0]))

    return fig, ax


# ── plot_lo_shu_grid ──────────────────────────────────────────────────────────

def plot_lo_shu_grid(
    hc         : "FLUHyperCell",
    perspective: Optional[object] = None,
    value_type : str               = "norm1",
    ax         : Optional["Axes"]  = None,
) -> Tuple["Figure", "Axes"]:
    """
    Annotated 9×9 grid showing a chosen value type for each cell.

    Parameters
    ----------
    hc          : FLUHyperCell
    perspective : Perspective | None   if given, temporarily applies it
    value_type  : str  one of 'norm1', 'norm0', 'balanced', 'unity'
                  (default 'norm1')
    ax          : Axes | None

    Returns
    -------
    (Figure, Axes)

    Raises
    ------
    ImportError   if matplotlib not installed
    ValueError    if value_type is unknown
    """
    _require_matplotlib()

    valid = {"norm1", "norm0", "balanced", "unity"}
    if value_type not in valid:
        raise ValueError(f"value_type must be one of {valid}, got {value_type!r}")

    # Build value grid
    grid = np.zeros((9, 9), dtype=float)
    for r in range(9):
        for c in range(9):
            cell = hc.cell(r, c)
            grid[r, c] = getattr(cell, value_type)

    cmap    = "RdBu_r" if value_type == "balanced" else "YlOrRd"
    title   = f"Lo Shu HyperCell — {value_type}"
    if perspective is not None:
        title += f"\nPerspective: {perspective}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.get_figure()

    vabs = max(abs(grid.min()), abs(grid.max()))
    if value_type == "balanced":
        im = ax.imshow(grid, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="equal")
    else:
        im = ax.imshow(grid, cmap=cmap, aspect="equal")

    # Annotate each cell
    for r in range(9):
        for c in range(9):
            val = grid[r, c]
            label = f"{val:.3f}" if value_type == "unity" else str(int(val))
            # Colour boundary for contrast
            normed = (val - grid.min()) / (grid.max() - grid.min() + 1e-9)
            color  = "white" if normed > 0.7 or normed < 0.3 else "black"
            if value_type == "balanced":
                normed_b = (val + vabs) / (2 * vabs + 1e-9)
                color    = "white" if normed_b < 0.3 or normed_b > 0.7 else "black"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Column (d2 − 1)")
    ax.set_ylabel("Row    (d1 − 1)")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))

    return fig, ax


# ── plot_bijection_path ───────────────────────────────────────────────────────

def plot_bijection_path(
    n          : int,
    d          : int               = 2,
    ax         : Optional["Axes"]  = None,
    show_arrows: bool              = True,
    alpha      : float             = 0.6,
) -> Tuple["Figure", "Axes"]:
    """
    Scatter plot of the FM-Dance bijection path for d=2 or d=3.

    Plots all n^d step indices as points in their FM-Dance coordinate space,
    connected in step-index order to show the traversal path.

    Parameters
    ----------
    n           : int   FM-Dance order (odd, ≥ 3)
    d           : int   dimensionality — 2 for 2D scatter, 3 for 3D scatter
    ax          : Axes | None   existing axes (ignored for d=3, new 3D axes created)
    show_arrows : bool  draw direction arrows along the path (d=2 only)
    alpha       : float point / line transparency

    Returns
    -------
    (Figure, Axes)

    Raises
    ------
    ImportError   if matplotlib not installed
    ValueError    if d not in {2, 3} or n is even
    """
    _require_matplotlib()

    from flu.core.fm_dance import index_to_coords, generate_fast
    from flu.utils.math_helpers import is_odd

    if not is_odd(n):
        raise ValueError(f"FM-Dance requires odd n, got n={n}")
    if d not in (2, 3):
        raise ValueError(f"plot_bijection_path supports d=2 or d=3, got d={d}")

    total  = n ** d
    coords = np.array([index_to_coords(k, n, d) for k in range(total)], dtype=float)

    if d == 2:
        xs, ys = coords[:, 0], coords[:, 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        # Path line
        ax.plot(xs, ys, "-", color="steelblue", alpha=alpha * 0.7, linewidth=0.8, zorder=1)

        # Coloured scatter: colour by step index
        sc = ax.scatter(xs, ys, c=np.arange(total), cmap="plasma",
                        s=max(20, 400 // total), zorder=2, alpha=alpha)
        plt.colorbar(sc, ax=ax, label="Step index k", fraction=0.046, pad=0.04)

        # Direction arrows (every few steps)
        if show_arrows and total > 1:
            step = max(1, total // 12)
            for k in range(0, total - 1, step):
                dx = float(xs[k + 1] - xs[k])
                dy = float(ys[k + 1] - ys[k])
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    ax.annotate("", xy=(xs[k + 1], ys[k + 1]),
                                xytext=(xs[k], ys[k]),
                                arrowprops=dict(arrowstyle="->", color="navy",
                                                lw=1.2, alpha=0.7))

        ax.set_title(f"FM-Dance Bijection Path  n={n}, d=2\n"
                     f"({total} steps)", fontsize=11, fontweight="bold")
        ax.set_xlabel("coord[0]")
        ax.set_ylabel("coord[1]")
        ax.set_xticks(range(-n // 2, n // 2 + 1))
        ax.set_yticks(range(-n // 2, n // 2 + 1))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")

    else:  # d == 3
        fig = plt.figure(figsize=(8, 7))
        ax3 = fig.add_subplot(111, projection="3d")

        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        ax3.plot(xs, ys, zs, "-", color="steelblue", alpha=alpha * 0.5, linewidth=0.7)
        sc = ax3.scatter(xs, ys, zs, c=np.arange(total), cmap="plasma",
                         s=max(15, 300 // total), alpha=alpha)
        fig.colorbar(sc, ax=ax3, label="Step index k", fraction=0.03, pad=0.1)
        ax3.set_title(f"FM-Dance Bijection Path  n={n}, d=3\n"
                      f"({total} steps)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("coord[0]")
        ax3.set_ylabel("coord[1]")
        ax3.set_zlabel("coord[2]")
        ax = ax3

    return fig, ax
