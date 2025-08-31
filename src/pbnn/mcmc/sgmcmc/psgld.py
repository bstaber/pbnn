# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

"""Public API for the Preconditioned Stochastic gradient Langevin Dynamics kernel."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.types import Array, PRNGKey

from pbnn.mcmc.sgmcmc.diffusions import preconditioned_overdamped_langevin

__all__ = ["init", "kernel"]


class pSGLDState(NamedTuple):
    step: int
    position: Array
    logprob_grad: Array
    square_avg: Array


def init(position: Array, batch, grad_estimator_fn: Callable):
    """Initialize the pSGLD state."""
    logprob_grad = grad_estimator_fn(position, batch)
    square_avg = jax.tree_util.tree_map(jnp.square, logprob_grad)
    return pSGLDState(0, position, logprob_grad, square_avg)


def kernel(grad_estimator_fn: Callable) -> Callable:
    """Builds a one-step pSGLD transition kernel."""
    integrator = preconditioned_overdamped_langevin(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey,
        state: pSGLDState,
        data_batch: Array,
        step_size: float,
        alpha: float,
    ) -> pSGLDState:
        step, *diffusion_state = state
        new_state = integrator(rng_key, diffusion_state, step_size, alpha, data_batch)

        return pSGLDState(step + 1, *new_state)

    return one_step
