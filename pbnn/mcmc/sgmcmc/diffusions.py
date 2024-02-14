from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.sgmcmc.diffusions import generate_gaussian_noise
from blackjax.types import Array, PRNGKey


class pSGLDState(NamedTuple):
    position: Array
    logprob_grad: Array
    square_avg: Array


def preconditioned_overdamped_langevin(logprob_grad_fn):
    """Euler solver for overdamped Langevin diffusion with preconditioning [0]_.

        References
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


class SGHMCCVState(NamedTuple):
    position: Array
    momentum: Array
    batch_logprob_grad: Array
    c_position: Array
    c_full_logprob_grad: Array
    c_batch_logprob_grad: Array


def sghmccv(logprob_grad_fn, alpha: float = 0.01, beta: float = 0):
    """Solver for the diffusion equation of the SGHMC algorithm with control variates.

    References
    ----------
    .. [0]:  Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2019).
             Control variates for stochastic gradient MCMC.
             Statistics and Computing, 29(3), 599-615.

    """

    def one_step(
        rng_key: PRNGKey, state: SGHMCCVState, step_size: float, batch: tuple = ()
    ) -> SGHMCCVState:
        (
            position,
            momentum,
            batch_logprob_grad,
            c_position,
            c_full_logprob_grad,
            c_batch_logprob_grad,
        ) = state

        logprob_grad = jax.tree_util.tree_map(
            lambda g0, g1, g2: g0 - g2 + g1,
            c_full_logprob_grad,
            batch_logprob_grad,
            c_batch_logprob_grad,
        )

        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda x, p: x + step_size * p, position, momentum
        )
        momentum = jax.tree_util.tree_map(
            lambda p, g, n: (1.0 - alpha * step_size) * p
            + step_size * g
            + jnp.sqrt(step_size * (2 * alpha - step_size * beta)) * n,
            momentum,
            logprob_grad,
            noise,
        )

        batch_logprob_grad = logprob_grad_fn(position, batch)
        c_batch_logprob_grad = logprob_grad_fn(c_position, batch)
        return SGHMCCVState(
            position,
            momentum,
            batch_logprob_grad,
            c_position,
            c_full_logprob_grad,
            c_batch_logprob_grad,
        )

    return one_step
