# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

from typing import Callable

import jax
import optax
from flax.training import train_state
from jax import Array


def create_train_state(rng, flax_module, init_input, learning_rate, optimizer="adam"):
    """Creates initial `TrainState`."""
    params = flax_module.init(rng, init_input)["params"]
    if optimizer == "sgd":
        tx = optax.sgd(learning_rate)
    elif optimizer == "adam":
        tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=flax_module.apply, params=params, tx=tx
    )


def train_fn(
    logposterior_fn: Callable,
    network: Callable,
    train_ds: dict,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    rng_key: Array,
    optimizer: str = "adam",
):
    """Function that estimates the maximum a posteriori given the log-posterior function provided by the user.

    Parameters
    ----------

    logposterior_fn
        Callable logposterior function
    network
        Neural network given as a flax.linen.nn
    train_ds
        Training dataset given as a dict {"x": X, "y": y}
    batch_size
        Batch size
    num_epochs
        Number of epochs
    learning_rate
        Value of the step size
    rng_key
        A random seed
    optimizer
        Chosen optimizer given as a string ("sgd", "adam")

    Returns
    -------

    Values of the parameters

    """

    @jax.jit
    def train_step(state, batch):
        """Train for a single step"""

        def loss_fn(params):
            loss = -logposterior_fn(params, (batch["x"], batch["y"]))
            return loss

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    def train_epoch(state, train_ds, batch_size, rng):
        """Train for a single epoch."""
        train_ds_size = len(train_ds["x"])
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        def one_step_fn(state, perm):
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            state = train_step(state, batch)
            return state, None

        state, _ = jax.lax.scan(one_step_fn, state, perms)
        return state

    rng_key, init_rng = jax.random.split(rng_key)

    initial_state = create_train_state(
        init_rng,
        network,
        train_ds["x"][:batch_size],
        learning_rate,
        optimizer=optimizer,
    )
    del init_rng

    @jax.jit
    def one_step(state, rng_key):
        state = train_epoch(state, train_ds, batch_size, rng_key)
        return state, state

    keys = jax.random.split(rng_key, num_epochs + 1)
    last_state, _ = jax.lax.scan(one_step, initial_state, keys)

    return last_state.params
