from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from pbnn.utils.misc import build_logposterior_estimator_fn, create_train_state


def logposterior_estimator_fn(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Log posterior function"""

    def logposterior_fn(parameters, data_batch, dropout_rng):
        logprior = logprior_fn(parameters)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, in_axes=(None, 0, None))
        return logprior + data_size * jnp.mean(
            batch_loglikelihood(parameters, data_batch, dropout_rng), axis=0
        )

    return logposterior_fn


def mcdropout_fn(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    network: nn.Module,
    batch_size: int,
    num_epochs: int,
    step_size: float,
    rng_key: Array
):
    data_size = len(X)
    train_ds = {"x": X, "y": y}
    logposterior_fn = build_logposterior_estimator_fn(
        logprior_fn, loglikelihood_fn, data_size
    )

    def loss_fn(params, batch_data, dropout_rng):
        return -logposterior_fn(params, batch_data, dropout_rng)

    grad_fn = jax.jit(jax.grad(loss_fn))

    def train_epoch(state, train_ds, batch_size, rng, dropout_rng):
        """Train for a single epoch."""
        train_ds_size = len(train_ds["x"])
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        def one_step_fn(state, xs):
            perm, dropout_key = xs
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            grads = grad_fn(state.params, (batch["x"], batch["y"]), dropout_key)
            state = state.apply_gradients(grads=grads)
            return state, None

        keys = jax.random.split(dropout_rng, steps_per_epoch)
        state, _ = jax.lax.scan(one_step_fn, state, (perms, keys))
        return state

    # Define the one epoch train function
    def one_train_epoch(state, keys):
        rng_key, dropout_rng_key = keys
        state = train_epoch(state, train_ds, batch_size, rng_key, dropout_rng_key)
        return state, state

    rng_key, dropout_rng_key, init_rng = jax.random.split(rng_key, num=3)

    perm_keys = jax.random.split(rng_key, num_epochs + 1)
    dropout_keys = jax.random.split(dropout_rng_key, num_epochs + 1)
    initial_state = create_train_state(
        init_rng, dropout_rng_key, network(), train_ds["x"][0], step_size
    )

    state, _ = jax.lax.scan(one_train_epoch, initial_state, (perm_keys, dropout_keys))

    def ravel_fn(pytree):
        return ravel_pytree(pytree)[0]

    def predict_fn(network, params, X_test, dropout_rng):
        return jax.vmap(
            lambda x: network().apply(
                {"params": params},
                x,
                rngs={"dropout": dropout_rng},
                deterministic=False,
            ),
            0,
        )(X_test)

    return state.params, ravel_fn, predict_fn
