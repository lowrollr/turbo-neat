from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp


def mask_data(x, y, mask):
    """shape a mask appropriately to broadcast over x and y, then apply it to x and y"""
    mask = mask.reshape((-1,) + (1,) * (x.ndim - 1))
    return jnp.where(mask, x, y)


def apply_where(
    tree: chex.ArrayTree, fn: Callable, mask: chex.Array, *args, **kwargs
) -> chex.ArrayTree:
    """Apply a vmapped function conditionally to a batched tree

    Additional arguments can be passed via args and kwargs:
    * args are vmapped
    * kwargs are not vmapped
    """
    mask_fn = partial(mask_data, mask=mask)
    partial_fn = partial(fn, **kwargs)
    vmapped_fn = jax.vmap(partial_fn)
    return jax.tree_map(mask_fn, vmapped_fn(tree, *args), tree)


def apply(tree: chex.ArrayTree, fn: Callable, *args, **kwargs) -> chex.ArrayTree:
    """Apply a function to a batch of genomes
     This will usually be an unbound method of Genome

    Additional arguments can be passed via args and kwargs:
     * args are vmapped
     * kwargs are not vmapped
    """
    partial_fn = partial(fn, **kwargs)
    vmapped_fn = jax.vmap(partial_fn)
    return vmapped_fn(tree, *args)


def round_to_integers(arr, target_sum):
    # Round the float values
    rounded = jnp.floor(arr).astype(int)
    # Calculate the remainder for each element
    remainder = arr - rounded
    # Calculate how many values need to be adjusted to preserve the sum
    adjustment_needed = target_sum - rounded.sum()
    # Get the indices of the largest remainders
    indices_to_adjust = jnp.argsort(remainder, descending=True)
    adjustment_mask = jnp.arange(arr.shape[0]) < adjustment_needed
    adjusted = rounded.at[indices_to_adjust].add(adjustment_mask)
    return adjusted


def is_printable(x):
    return (
        isinstance(x, float)
        or isinstance(x, int)
        or isinstance(x, str)
        or isinstance(x, bool)
        or isinstance(x, jnp.ndarray)
    )
