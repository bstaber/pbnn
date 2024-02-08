from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from pbnn.utils.misc import build_logposterior_estimator_fn, create_train_state


def deep_ensembles_fn(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    network: nn.Module,
    batch_size: int,
    num_epochs: int,
    step_size: float,
    num_networks: int,
    rng_key: Array,
):
    """Functions that performs deep ensembles.

    Parameters
    ----------

    X
        Matrix of input features of size (N, d)
    y
        Matrix of output features of size (N, s)
    loglikelihood_fn
        Callable loglikelihood function
    logprior_fn
        Callable logprior function
    network
        Neural network given as a flax.linen.Module
    batch_size
        Batch size
    num_epochs
        Number of epochs used to train each member
    step_size
        Value of the step size (learning rate)
    num_networks
        Number of members in the deep ensembles
    rng_key
        A random seed

    Returns
    -------

    parameters
        Parameters of all the networks given as a PyTree
    ravel_fn
        Function that flattens the networks parameters
    predict_fn
        Function that makes predictions using the deep ensembles

    """
    data_size = len(X)
    train_ds = {"x": X, "y": y}

    logposterior_fn = build_logposterior_estimator_fn(
        logprior_fn, loglikelihood_fn, data_size
    )

    def loss_fn(params, batch_data):
        return -logposterior_fn(params, batch_data)

    grad_fn = jax.jit(jax.grad(loss_fn))

    def train_epoch(state, train_ds, batch_size, rng):
        """Train for a single epoch."""
        train_ds_size = len(train_ds["x"])
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        def one_step_fn(state, perm):
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            grads = grad_fn(state.params, (batch["x"], batch["y"]))
            state = state.apply_gradients(grads=grads)
            return state, None

        state, _ = jax.lax.scan(one_step_fn, state, perms)
        return state

    # Define the one epoch train function
    def one_train_epoch(state, rng_key):
        state = train_epoch(state, train_ds, batch_size, rng_key)
        return state, state

    # Define one training function
    def one_training(carry, idx):
        initial_state, rng_key = carry
        keys = jax.random.split(rng_key, num_epochs + 1)
        state, _ = jax.lax.scan(one_train_epoch, initial_state, keys)

        rng_key = jax.random.PRNGKey(idx + 1)
        rng_key, init_rng = jax.random.split(rng_key)

        params = network().init(init_rng, train_ds["x"][0])["params"]
        initial_state.replace(params=params)
        return (initial_state, rng_key), state

    rng_key, init_rng = jax.random.split(rng_key)
    initial_state = create_train_state(init_rng, network(), train_ds["x"][0], step_size)
    _, states = jax.lax.scan(
        one_training, (initial_state, rng_key), jnp.arange(0, num_networks)
    )

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return states.params, ravel_fn, predict_fn
