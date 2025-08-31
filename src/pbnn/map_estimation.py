"""Module for MAP estimation using neural networks using JAX and Flax."""
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import Array

from pbnn.utils.data import NumpyDataset, NumpyLoader


def create_train_state(
    rng: Array,
    flax_module: nn.Module,
    init_input: Array,
    learning_rate: float,
    optimizer: Optional[str] = "adam",
) -> train_state.TrainState:
    """Creates initial `TrainState`.

    Args:
        rng: Initial random key
        flax_module: Flax module to use
        init_input: Initial input to the module
        learning_rate: Learning rate to use
        optimizer: Optimizer to use. Defaults to "adam"

    Returns: train_state.TrainState: initial training state.
    """
    params = flax_module.init(rng, init_input)["params"]
    tx = optax.adam(learning_rate) if optimizer == "adam" else optax.sgd(learning_rate)
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

    Args:
        logposterior_fn: Callable logposterior function
        network: Neural network given as a flax.linen.nn
        train_ds: Training dataset given as a dict {"x": X, "y": y}
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Value of the step size
        rng_key: A random seed
        optimizer: Chosen optimizer given as a string ("sgd", "adam")

    Returns: MAP estimation of the parameters.
    """

    def loss_fn(params, batch):
        return -logposterior_fn(params, batch)

    grad_fn = jax.grad(loss_fn)

    def train_step(state, batch):
        grads = grad_fn(state.params, batch)
        return state.apply_gradients(grads=grads)

    @jax.jit
    def run_epoch(state, batches):
        def step_fn(state, batch):
            state = train_step(state, batch)
            return state, None

        state, _ = jax.lax.scan(step_fn, state, batches)
        return state

    rng_key, init_rng = jax.random.split(rng_key)
    initial_state = create_train_state(
        init_rng,
        network,
        train_ds["x"][:batch_size],
        learning_rate,
        optimizer=optimizer,
    )

    # Prebatch data to allow scan
    dataset = NumpyDataset(train_ds["x"], train_ds["y"])
    data_loader = NumpyLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    batches = list(data_loader)
    batches = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *batches
    )  # shape: (num_batches, ...)

    state = initial_state
    for _ in range(num_epochs):
        state = run_epoch(state, batches)

    return state.params
