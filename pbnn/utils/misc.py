from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


def create_train_state(rng, flax_module, init_input, learning_rate):
    """Creates initial `TrainState`."""
    # model = flax_module()
    params = flax_module.init(rng, init_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=flax_module.apply, params=params, tx=tx
    )


def build_logposterior_estimator_fn(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Log posterior function"""

    def logposterior_fn(parameters, data_batch):
        logprior = logprior_fn(parameters)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, in_axes=(None, 0))
        return logprior + data_size * jnp.mean(
            batch_loglikelihood(parameters, data_batch), axis=0
        )

    return logposterior_fn
