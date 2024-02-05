"""Public API for the Stochastic gradient Langevin Dynamics kernel with control variates."""
from typing import Callable, NamedTuple

import jax
from blackjax.sgmcmc.diffusions import overdamped_langevin
from blackjax.types import PRNGKey, PyTree

__all__ = ["SGLDCVState", "init", "kernel"]


class SGLDCVState(NamedTuple):
    step: int
    position: PyTree
    batch_logprob_grad: PyTree
    c_position: PyTree
    c_full_logprob_grad: PyTree
    c_batch_logprob_grad: PyTree


def init(
    position: PyTree,
    c_position: PyTree,
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


def kernel(grad_estimator_fn: Callable) -> Callable:
    integrator = overdamped_langevin(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey, state: SGLDCVState, data_batch: PyTree, step_size: float
    ) -> SGLDCVState:
        step, *diffusion_state = state
        (
            position,
            batch_logprob_grad,
            c_position,
            c_full_logprob_grad,
            c_batch_logprob_grad,
        ) = diffusion_state

        logprob_grad = jax.tree_util.tree_map(
            lambda g0, g1, g2: g0 - g2 + g1,
            c_full_logprob_grad,
            batch_logprob_grad,
            c_batch_logprob_grad,
        )

        new_state = integrator(rng_key, position, logprob_grad, step_size, data_batch)
        c_batch_logprob_grad = grad_estimator_fn(c_position, data_batch)

        return SGLDCVState(
            step + 1,
            new_state.position,
            new_state.logprob_grad,
            c_position,
            c_full_logprob_grad,
            c_batch_logprob_grad,
        )

    return one_step
