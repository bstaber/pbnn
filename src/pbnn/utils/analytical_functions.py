"""Analytical test functions with noise for regression tasks."""
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import jax.numpy as jnp
from jax import Array


def trigonometric_function(x: Array, noise: Array) -> Array:
    """Simple trigonometric function with homoscedastic noise."""
    f = jnp.cos(2.0 * x) + jnp.sin(x)
    y = f + noise
    return f, y


def barber_fn(x: Array, noise: Array) -> Array:
    """Simple linear function with heteroscedastic noise."""
    f = x / 2.0
    y = x / 2.0 + noise * jnp.abs(jnp.sin(x))
    return f, y


def gramacy_function(x: Array, noise: Array) -> Array:
    """Non-stationnary test function from Gramacy (2007) with additive homoscedastic noise."""
    idx_1 = x[:, 0] <= 9.6
    idx_2 = x[:, 0] > 9.6
    f = jnp.zeros_like(x)
    f = f.at[idx_1].set(
        jnp.sin(jnp.pi * x[idx_1] / 5.0)
        + (1.0 / 5.0) * jnp.cos(4.0 * jnp.pi * x[idx_1] / 5.0)
    )
    f = f.at[idx_2].set(x[idx_2] / 10.0 - 1.0)
    y = f + noise
    return f, y
