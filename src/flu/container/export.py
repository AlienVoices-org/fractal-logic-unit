"""
flu/container/export.py
=======================
Zero-copy buffer export from SparseCommunionManifold to ML framework tensors.

Fills a caller-provided, pre-allocated buffer directly — no intermediate
np.ndarray materialised for the full n^D array.  Derived from T1 (O(D)
addressing) and the additive separability of the communion sum.

Supported targets
-----------------
  to_numpy_buffer(M, out)            — fills a NumPy ndarray in-place
  to_torch_buffer(M, out)            — fills a torch.Tensor in-place
  to_jax_buffer(M, coords_nd)        — returns a JAX array (JAX is immutable)
  fill_weight_matrix(M, rows, cols)  — convenience: 2-D weight matrix [rows×cols]

Design principles
-----------------
* Dependencies are OPTIONAL — torch and jax are imported lazily; the module
  loads fine without either installed.
* Batched vectorised evaluation via SparseCommunionManifold._batch_evaluate()
  keeps the Python loop over cells to zero.
* Signed float normalisation (divide by ⌊n/2⌋) maps the integer range
  [−⌊n/2⌋, ⌊n/2⌋] to [−1, 1] — ready for use as neural weights (S2-GAUSS
  guarantees spectral flatness, S1 guarantees zero mean).

STATUS: DESIGN INTENT — correctness follows from SparseCommunionManifold;
        framework-specific type rules are the responsibility of the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from flu.container.sparse import SparseCommunionManifold


# ── helpers ───────────────────────────────────────────────────────────────────

def _all_coords(n: int, d: int) -> np.ndarray:
    """
    Build an (n^D × D) array of all signed coordinates in FM-Dance order.

    Each row is one coordinate tuple (x_0, …, x_{D-1}) with x_i ∈ [−⌊n/2⌋, ⌊n/2⌋].
    Constructed via NumPy meshgrid — no Python loop, no full value array.
    """
    half   = n // 2
    axis   = np.arange(-half, -half + n, dtype=np.int32)
    grids  = np.meshgrid(*([axis] * d), indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=-1)  # (n^D, D)
    return coords


def _normalise(values: np.ndarray, n: int, dtype=np.float32) -> np.ndarray:
    """
    Map integer values in [−⌊n/2⌋, ⌊n/2⌋] → float in [−1.0, 1.0].

    Division by ⌊n/2⌋ guarantees:
      • mean = 0  (S1 — zero global mean preserved)
      • max |value| = 1.0
    """
    half = max(n // 2, 1)
    return (values.astype(dtype)) / dtype(half)


# ── public API ────────────────────────────────────────────────────────────────

def to_numpy_buffer(
    manifold: "SparseCommunionManifold",
    out: Optional[np.ndarray] = None,
    normalise: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    Fill (or create) a NumPy ndarray from a SparseCommunionManifold.

    No intermediate full-array allocation: values are computed in a single
    vectorised batch call, then written directly into ``out``.

    Parameters
    ----------
    manifold  : SparseCommunionManifold
    out       : pre-allocated ndarray of shape manifold.shape, or None
                (a new array is allocated if None).
    normalise : if True, divide by ⌊n/2⌋ so values ∈ [−1, 1].
    dtype     : NumPy dtype for the output buffer (default float32).

    Returns
    -------
    np.ndarray  shape manifold.shape

    Raises
    ------
    ValueError  if out.shape != manifold.shape
    """
    if out is not None and out.shape != manifold.shape:
        raise ValueError(
            f"out.shape {out.shape} != manifold.shape {manifold.shape}."
        )

    coords = _all_coords(manifold.n, manifold.d)          # (n^D, D)
    flat   = manifold._batch_evaluate(coords).astype(dtype)  # (n^D,)
    if normalise:
        flat = _normalise(flat, manifold.n, dtype)

    shaped = flat.reshape(manifold.shape)
    if out is None:
        return shaped.copy()
    np.copyto(out, shaped.astype(out.dtype))
    return out


def to_torch_buffer(
    manifold: "SparseCommunionManifold",
    out=None,   # torch.Tensor | None
    normalise: bool = True,
    dtype=None,  # torch.dtype | None  (default: torch.float32)
):
    """
    Fill a pre-allocated ``torch.Tensor`` from a SparseCommunionManifold.

    Skips the np.ndarray bottleneck by writing directly into the tensor's
    data buffer via ``torch.Tensor.copy_()``.  The intermediate NumPy array
    is only the flat batch result (n^D values × 4 bytes), not the full shaped
    array — it is immediately discarded after the copy.

    Parameters
    ----------
    manifold  : SparseCommunionManifold
    out       : pre-allocated torch.Tensor of shape manifold.shape, or None.
                If None, a new CPU tensor is created.
    normalise : divide by ⌊n/2⌋ → float range [−1, 1].
    dtype     : torch.dtype for the output (default torch.float32).

    Returns
    -------
    torch.Tensor  shape manifold.shape

    Raises
    ------
    ImportError  if torch is not installed.
    ValueError   if out.shape != manifold.shape.

    Example
    -------
    >>> import torch
    >>> from flu.container.sparse import SparseCommunionManifold
    >>> from flu.container.export import to_torch_buffer
    >>> from flu.core.factoradic import unrank_optimal_seed
    >>> seeds = [unrank_optimal_seed(0, 5, signed=False)] * 3
    >>> M = SparseCommunionManifold(n=5, seeds=seeds)
    >>> buf = torch.empty(M.shape)
    >>> to_torch_buffer(M, out=buf)   # fills in-place, returns buf
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is not installed. "
            "Install it with: pip install torch"
        ) from e

    if dtype is None:
        dtype = torch.float32

    if out is not None and tuple(out.shape) != manifold.shape:
        raise ValueError(
            f"out.shape {tuple(out.shape)} != manifold.shape {manifold.shape}."
        )

    # Compute flat batch values on CPU as NumPy
    coords   = _all_coords(manifold.n, manifold.d)
    flat_np  = manifold._batch_evaluate(coords).astype(np.float32)
    if normalise:
        flat_np = _normalise(flat_np, manifold.n, np.float32)
    shaped_np = flat_np.reshape(manifold.shape)

    if out is None:
        out = torch.from_numpy(shaped_np).to(dtype)
    else:
        # Copy without creating a full Python-side tensor duplicate:
        # from_numpy shares memory until we call .to(dtype) or .copy_()
        tmp = torch.from_numpy(shaped_np)
        out.copy_(tmp.to(dtype))

    return out


def to_jax_buffer(
    manifold: "SparseCommunionManifold",
    normalise: bool = True,
    dtype=None,  # jax.numpy dtype | None  (default: jnp.float32)
):
    """
    Create a JAX array from a SparseCommunionManifold.

    JAX arrays are immutable, so this function returns a *new* array rather
    than filling in-place.  The intermediate computation path remains
    identical: batch evaluation → normalise → reshape → jnp.asarray.

    Parameters
    ----------
    manifold  : SparseCommunionManifold
    normalise : divide by ⌊n/2⌋ → float range [−1, 1].
    dtype     : jax.numpy dtype (default jnp.float32).

    Returns
    -------
    jax.numpy.ndarray  shape manifold.shape

    Raises
    ------
    ImportError  if jax is not installed.

    Example
    -------
    >>> from flu.container.export import to_jax_buffer
    >>> arr = to_jax_buffer(M)   # shape M.shape, values in [−1, 1]
    """
    try:
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError(
            "jax is not installed. "
            "Install it with: pip install jax"
        ) from e

    if dtype is None:
        dtype = jnp.float32

    coords  = _all_coords(manifold.n, manifold.d)
    flat_np = manifold._batch_evaluate(coords).astype(np.float32)
    if normalise:
        flat_np = _normalise(flat_np, manifold.n, np.float32)

    return jnp.asarray(flat_np.reshape(manifold.shape), dtype=dtype)


def fill_weight_matrix(
    manifold: "SparseCommunionManifold",
    rows: int,
    cols: int,
    normalise: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convenience function: extract a 2-D weight matrix [rows × cols] from the
    manifold, sampling cells in FM-Dance rank order.

    Useful for neural layer initialisation when n^D >> rows*cols (subsample)
    or when n^D == rows*cols (exact fit).

    Parameters
    ----------
    manifold  : SparseCommunionManifold
    rows, cols: weight matrix shape
    normalise : divide by ⌊n/2⌋ → float range [−1, 1].
    dtype     : NumPy dtype.

    Returns
    -------
    np.ndarray  shape (rows, cols)

    Raises
    ------
    ValueError  if rows * cols > n^D (not enough cells).
    """
    total = manifold.n ** manifold.d
    needed = rows * cols
    if needed > total:
        raise ValueError(
            f"Requested {rows}×{cols}={needed} cells but manifold has only "
            f"n^D={total} cells (n={manifold.n}, d={manifold.d})."
        )

    from flu.core.fm_dance import index_to_coords

    # Build (needed, D) signed coordinate array in FM-Dance rank order
    coords = np.array(
        [list(index_to_coords(k, manifold.n, manifold.d)) for k in range(needed)],
        dtype=np.int32,
    )
    flat = manifold._batch_evaluate(coords).astype(dtype)
    if normalise:
        flat = _normalise(flat, manifold.n, dtype)
    return flat.reshape(rows, cols)
