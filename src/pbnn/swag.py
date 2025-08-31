"""SWAG algorithm for Bayesian neural networks."""
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from typing import Callable, NamedTuple, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import Array
from jax.flatten_util import ravel_pytree

from pbnn.utils.misc import build_logposterior_estimator_fn


class SWAGState(NamedTuple):
    """Named tuple for the SWAG state.

    Attributes:
        num_models: int
        mean: flax.core.FrozenDict
        som: flax.core.FrozenDict
    """

    num_models: int
    mean: flax.core.FrozenDict
    som: flax.core.FrozenDict


class SwagModel:
    """Builds a SWAG model."""

    def __new__(
        cls, logposterior_fn: Callable, train_ds: dict, batch_size: int, initial_state
    ):
        """Creates a function that estimates the maximum a posteriori given the log-posterior function provided by the user.

        Args:
            logposterior_fn: Callable logposterior function
            train_ds: Training dataset given as a dict {"x": X, "y": y}
            batch_size: Batch size
            initial_state: Initial state
        """

        def loss_fn(params, batch):
            loss = -logposterior_fn(params, (batch["x"], batch["y"]))
            return loss

        grad_fn = jax.jit(jax.grad(loss_fn))

        def train_epoch(state, swag_state, train_ds, batch_size, rng):
            """Train for a single epoch."""
            train_ds_size = len(train_ds["x"])
            steps_per_epoch = train_ds_size // batch_size

            perms = jax.random.permutation(rng, train_ds_size)
            perms = perms[: steps_per_epoch * batch_size]
            perms = perms.reshape((steps_per_epoch, batch_size))

            # for perm in perms:
            def one_step_fn(state, perm):
                batch = {k: v[perm, ...] for k, v in train_ds.items()}
                grads = grad_fn(state.params, batch)
                state = state.apply_gradients(grads=grads)
                return state, None

            state, _ = jax.lax.scan(one_step_fn, state, perms)

            num_models = swag_state.num_models
            mean = swag_state.mean
            som = swag_state.som

            mean = jax.tree_util.tree_map(
                lambda x, data: x * num_models / (num_models + 1.0)
                + data / (num_models + 1.0),
                mean,
                state.params,
            )
            som = jax.tree_util.tree_map(
                lambda x, data: x * num_models / (num_models + 1.0)
                + jnp.square(data) / (num_models + 1.0),
                som,
                state.params,
            )
            return state, SWAGState(num_models + 1, mean, som)

        def make_train_step_fn(with_cov: bool):
            @jax.jit
            def train_step(states, rng_key):
                train_state, swag_state = states
                train_state, swag_state = train_epoch(
                    train_state, swag_state, train_ds, batch_size, rng_key
                )
                if with_cov:
                    dev = jax.tree_util.tree_map(
                        lambda x, y: x - y, train_state.params, swag_state.mean
                    )
                    return (train_state, swag_state), dev
                else:
                    return (train_state, swag_state), None

            return train_step

        train_step_wo_cov = make_train_step_fn(with_cov=False)
        train_step_with_cov = make_train_step_fn(with_cov=True)

        def train_fn(num_epochs: int, rank: int, rng_key: Array):
            # rng_key = jax.random.PRNGKey(0)
            rng_key_wo_cov, rng_key_w_cov = jax.random.split(rng_key)

            keys = jax.random.split(rng_key_wo_cov, num_epochs - rank)
            initial_swag_state = SWAGState(
                1,
                initial_state.params,
                jax.tree_util.tree_map(lambda x: jnp.square(x), initial_state.params),
            )

            (train_state, swag_state), _ = jax.lax.scan(
                train_step_wo_cov, (initial_state, initial_swag_state), keys
            )

            keys = jax.random.split(rng_key_w_cov, rank)
            (train_state, swag_state), devs = jax.lax.scan(
                train_step_with_cov, (train_state, swag_state), keys
            )

            return (train_state, swag_state), devs

        return train_fn


def swag_fn(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    network: nn.Module,
    init_positions: Array,
    batch_size: int,
    num_epochs: int,
    step_size: float,
    cov_rank: int,
    rng_key: Array,
) -> Tuple[Array, Callable, Callable]:
    """Function that performs the SWAG algorithm.

    Args:
        X: Matrix of input features (N, d)
        y: Matrix of output features (N, s)
        loglikelihood_fn: Callable loglikelihood function
        logprior_fn: Callable logprior function
        network: Neural network given as a flax.linen.nn
        init_positions: Initial parameters of the network
        batch_size: Batch size
        num_epochs: Number of epochs
        step_size: Value of the step size
        cov_rank: Rank of the covariance approximation in the SWAG method
        rng_key: A random seed

    Returns: Parameters of the obtained SWAG model, a function that flattens the parameters and a function that makes predictions using the SWAG model.
    """
    data_size = len(X)
    train_ds = {"x": X, "y": y}
    logposterior_fn = build_logposterior_estimator_fn(
        logprior_fn, loglikelihood_fn, data_size
    )

    flat_init_positions, unravel_fn = ravel_pytree(init_positions)
    num_params = len(flat_init_positions)
    keys = jax.random.split(rng_key)

    num_samples = 2000
    z1 = jax.random.normal(keys[0], shape=(num_samples, num_params))
    z2 = jax.random.normal(keys[1], shape=(num_samples, cov_rank))

    init_state = train_state.TrainState.create(
        apply_fn=network.apply, params=init_positions, tx=optax.sgd(step_size)
    )

    swag_train_fn = SwagModel(logposterior_fn, train_ds, batch_size, init_state)

    _, key = jax.random.split(rng_key)
    (_, swag_state), deviations = swag_train_fn(
        num_epochs, cov_rank, jax.random.PRNGKey(0)
    )

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    Dvec = ravel_fn(deviations)
    mean_swa, unravel_fn = ravel_pytree(swag_state.mean)
    som_swa, _ = ravel_pytree(swag_state.som)
    SigDiag = jnp.diag(jnp.sqrt(jnp.clip(som_swa - mean_swa**2, a_min=0.0)))

    params = (
        mean_swa[None, :]
        + (1.0 / jnp.sqrt(2.0)) * jnp.matmul(z1, SigDiag)
        + (1.0 / jnp.sqrt(2.0 * (cov_rank - 1.0))) * jnp.matmul(z2, Dvec)
    )
    params_dict = jax.vmap(unravel_fn, 0)(params)

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network.apply({"params": p}, X_test), 0)(
            params
        ).squeeze()

    return params_dict, ravel_fn, predict_fn
