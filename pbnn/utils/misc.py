# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import Array


def create_train_state(rng, flax_module, init_input, learning_rate):
    """Creates an initial Flax `TrainState`.

    Parameters
    ----------

    rng
        Random seed key
    flax_module
        A Flax Module such as a network
    init_input
        Arbitrary input features used to instantiate initial network parameters
    learning_rate
        Step size

    Returns
    -------

    An initial train state

    """
    params = flax_module.init(rng, init_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=flax_module.apply, params=params, tx=tx
    )


def build_logposterior_estimator_fn(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Builds a callable logposterior function.

    Parameters
    ----------

    logprior_fn
        Callable logprior function
    loglikelihood_fn
        Callable loglikelihood function
    data_size
        Dataset size

    Returns
    -------

    Callable logposterior function

    """

    def logposterior_fn(parameters, data_batch):
        logprior = logprior_fn(parameters)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, in_axes=(None, 0))
        return logprior + data_size * jnp.mean(
            batch_loglikelihood(parameters, data_batch), axis=0
        )

    return logposterior_fn


def thinning_fn(positions: Array, size: int):
    """Thins MCMC outputs by greedily minimizing the energy distance between
    empirical distributions.

    Parameters
    ----------

    positions
        MCMC chain given as an array of size (N, d)
    size
        Subsampling size

    Returns
    -------

    Array of indices corresponding to the selected particles

    """

    def kernelfn(x1, x2):
        return jnp.linalg.norm(x1) + jnp.linalg.norm(x2) - jnp.linalg.norm(x1 - x2)

    kmap = jax.jit(jax.vmap(kernelfn, (None, 0)))
    k0_diag = jax.vmap(lambda y: 2.0 * jnp.linalg.norm(y))

    @jax.jit
    def GramDistKernel(X: Array, Y: Array):
        return jax.vmap(
            lambda x: jax.vmap(
                lambda y: jnp.sqrt(x @ x)
                + jnp.sqrt(y @ y)
                - jnp.sqrt(jnp.clip(x @ x + y @ y - 2 * x @ y, a_min=0))
            )(Y)
        )(X)

    Kmat = GramDistKernel(positions, positions)
    k0_mean = jnp.mean(Kmat, axis=1)
    k0 = k0_diag(positions)
    obj = k0 - 2.0 * k0_mean
    init = jnp.argmin(obj)

    def thinning_step_fn(carry, xs):
        idx, obj = carry
        ki = kmap(positions[idx], positions)
        obj = obj + 2.0 * ki - 2.0 * k0_mean
        new_idx = jnp.argmin(obj)
        return (new_idx, obj), new_idx

    _, idx = jax.lax.scan(thinning_step_fn, (init, obj), jnp.arange(1, size))
    idx = jnp.append(idx, init)

    return idx
