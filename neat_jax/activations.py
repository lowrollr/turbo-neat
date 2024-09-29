from typing import Callable, List

import chex
import jax
import jax.numpy as jnp

ActivationSelector = Callable[[chex.Array, chex.Array], chex.Array]
ActivationFn = Callable[[chex.Array], chex.Array]


def make_activation_selector_fn(activation_fns: List[ActivationFn]):
    return lambda i, x: jax.lax.switch(i, activation_fns, x)


def iden(x: chex.Array) -> chex.Array:
    return x


def inv(x: chex.Array) -> chex.Array:
    return 1 / (x + 1e-8)


relu = jax.nn.relu
sigmoid = jax.nn.sigmoid
tanh = jax.nn.tanh
absl = jnp.abs
sin = jnp.sin
cos = jnp.cos
exp = jnp.exp

visualization_color_mapping = {
    "iden": "webmaroon",
    "relu": "midnightblue",
    "sigmoid": "darkorchid4",
    "tanh": "darkgreen",
    "inv": "darkolivegreen",
    "absolute": "darkslategray",
    "sin": "firebrick4",
    "cos": "olive",
    "exp": "darkorange4",
}
