from typing import Callable

import blackjax
import jax
from jax import Array


def inference_loop(rng_key, step_fn, initial_state, num_samples):
    """Inference loop with lax.scan"""

    @jax.jit
    def one_step(state, rng_key):
        state, info = step_fn(rng_key, state)
        return state, (info, state)

    keys = jax.random.split(rng_key, num_samples)
    _, (infos, states) = jax.lax.scan(one_step, initial_state, keys)
    return infos, states


def hmc(
    logprob_fn: Callable,
    init_positions: Array,
    num_samples: int,
    step_size: float,
    inverse_mass_matrix: Array,
    num_integration_steps: int,
    rng_key: Array,
):
    """Wrapper of the HMC algorithm implemented in BlackJAX.

    Parameters
    ----------

    logprob_fn
        Callable function that returns the log-probability of the target
    init_positions
        Initial guess for the HMC algorithm
    num_samples
        Number of iterations (burn-in included)
    step_size
        Value of the step size
    inverse_mass_matrix
        Flattened inverse mass matrix
    num_integration_steps
        Number of integration steps
    rng_key
        A JAX PRNGKey.

    Returns
    -------

    A tuple of states and informations
    """

    kern = blackjax.hmc(
        logprob_fn, step_size, inverse_mass_matrix, num_integration_steps
    )
    step_fn = jax.jit(kern.step)
    init_state = kern.init(init_positions)
    infos, states = inference_loop(rng_key, step_fn, init_state, num_samples)
    return infos, states
