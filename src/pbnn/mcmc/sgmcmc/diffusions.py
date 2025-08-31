"""Diffusion processes for SGMCMC algorithms."""
# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.sgmcmc.diffusions import generate_gaussian_noise
from blackjax.types import Array, PRNGKey


class pSGLDState(NamedTuple):
    """State of the pSGLD diffusion process."""

    position: Array
    logprob_grad: Array
    square_avg: Array


def preconditioned_overdamped_langevin(logprob_grad_fn):
    """Euler solver for overdamped Langevin diffusion with preconditioning [0]_.

    References:
    ----------
    .. [0]:  Li, C., Chen, C., Carlson, D., & Carin, L. (2016, February).
             Preconditioned stochastic gradient Langevin dynamics for deep neural networks.
             In Thirtieth AAAI Conference on Artificial Intelligence.

    """

    def one_step(
        rng_key: PRNGKey,
        state: pSGLDState,
        step_size: float,
        alpha: float,
        batch: tuple = (),
    ):
        position, logprob_grad, square_avg = state

        preconditioner = jax.tree_util.tree_map(
            lambda tree: 1.0 / (jnp.sqrt(tree) + 1e-6), square_avg
        )
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, d, g, n: p + step_size * d * g + jnp.sqrt(2 * d * step_size) * n,
            position,
            preconditioner,
            logprob_grad,
            noise,
        )

        logprob_grad = logprob_grad_fn(position, batch)
        square_avg = jax.tree_util.tree_map(
            lambda a, b: alpha * a + (1.0 - alpha) * jnp.square(b),
            square_avg,
            logprob_grad,
        )
        return pSGLDState(position, logprob_grad, square_avg)

    return one_step
