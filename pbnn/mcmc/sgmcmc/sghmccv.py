"""Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel with control variates"""
from typing import Callable

import jax

from pbnn.mcmc.sgmcmc.diffusions import SGHMCCVState
from pbnn.mcmc.sgmcmc.diffusions import sghmccv as sghmccv_diffusion
from blackjax.util import generate_gaussian_noise as sample_momentum
from blackjax.types import PRNGKey, Array

from typing import NamedTuple

__all__ = ["SGLDCVState", "init", "kernel"]


class SGLDCVState(NamedTuple):
    step: int
    position: Array
    batch_logprob_grad: Array
    c_position: Array
    c_full_logprob_grad: Array
    c_batch_logprob_grad: Array


def init(
    position: Array,
    c_position: Array,
    c_full_logprob_grad,
    batch,
    grad_estimator_fn: Callable,
) -> SGLDCVState:
    c_batch_logprob_grad = grad_estimator_fn(c_position, batch)
    batch_logprob_grad = grad_estimator_fn(position, batch)
    return SGLDCVState(
        0,
        position,
        batch_logprob_grad,
        c_position,
        c_full_logprob_grad,
        c_batch_logprob_grad,
    )


def kernel(
    grad_estimator_fn: Callable, alpha: float = 0.01, beta: float = 0
) -> Callable:
    integrator = sghmccv_diffusion(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey,
        state: SGLDCVState,
        data_batch: Array,
        step_size: float,
        L: int,
    ) -> SGLDCVState:
        step, *diffusion_state = state
        (
            position,
            batch_logprob_grad,
            c_position,
            c_full_logprob_grad,
            c_batch_logprob_grad,
        ) = diffusion_state
        momentum = sample_momentum(rng_key, position)
        diffusion_state = SGHMCCVState(
            position,
            momentum,
            batch_logprob_grad,
            c_position,
            c_full_logprob_grad,
            c_batch_logprob_grad,
        )

        def body_fn(state, rng_key):
            new_state = integrator(rng_key, state, step_size, data_batch)
            return new_state, new_state

        keys = jax.random.split(rng_key, L)
        last_state, _ = jax.lax.scan(body_fn, diffusion_state, keys)

        return SGLDCVState(
            step + 1,
            last_state.position,
            last_state.batch_logprob_grad,
            last_state.c_position,
            last_state.c_full_logprob_grad,
            last_state.c_batch_logprob_grad,
        )

    return one_step
