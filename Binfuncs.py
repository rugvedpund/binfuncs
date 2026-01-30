"""
binfunctions.py
tiny library to bin arrays
"""

import jax.numpy as jnp
from jax import Array


def binmasks(array: Array, bins: Array | list) -> Array:
    """x, bins -> masks(nbins, nx)"""
    bins = jnp.array(bins)
    nbins = bins.shape[0] - 1
    idxs = (
        jnp.searchsorted(bins, array, side="right") - 1
    )  # tries to insert x into bins, and returns indices
    idxs = jnp.clip(idxs, 0, nbins - 1).astype(jnp.int32)
    masks = idxs[None, :] == jnp.arange(nbins)[:, None]  # (nbins, x.size)
    return masks


def binidxs(array: Array, bins: Array | list) -> list[tuple]:
    masks = binmasks(array, bins)
    bidxs = [jnp.where(m) for m in masks]
    return bidxs


def bincounts(array, bins):
    bidxs = binmasks(array, bins)
    return [len(bi) for bi in bidxs]
