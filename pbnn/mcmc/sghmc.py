import jax
import jax.numpy as jnp
import blackjax
import kernels
from jax import Array

from jax.flatten_util import ravel_pytree
from pbnn.mcmc.sgmcmc.gradients import cv_grad_estimator
from blackjax.gradient import grad_estimator
from pbnn.utils.data import batch_data
from typing import Callable


def SGHMC(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    num_integration_steps: int,
    rng_key: Array,
):
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sghmc functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    kern = blackjax.sghmc(grad_fn, num_integration_steps)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGHMC with lax.scan
    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        new_state = step_fn(rng_key, state, batch, step_size)
        return new_state, new_state

    keys = jax.random.split(rng_key, num_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    return positions, ravel_fn


def AdaptiveSGHMC(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    schedule_fn: Callable,
    num_iterations: int,
    num_integration_steps: int,
    rng_key: Array,
):
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sghmc functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    kern = blackjax.sghmc(grad_fn, num_integration_steps)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGHMC with lax.scan
    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        step_size = schedule_fn(state.step)
        new_state = step_fn(rng_key, state, batch, step_size)
        return new_state, new_state

    keys = jax.random.split(rng_key, num_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    return positions, ravel_fn


def SGHMCCV(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    centering_positions: Array,
    num_integration_steps: int,
    rng_key: Array,
):
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sghmc functions
    grad_fn = cv_grad_estimator(
        logprior_fn, loglikelihood_fn, (X, y), centering_positions
    )

    kern = blackjax.sghmc(grad_fn, num_integration_steps)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGHMC with lax.scan
    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        new_state = step_fn(rng_key, state, batch, step_size)
        return new_state, new_state

    keys = jax.random.split(rng_key, num_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    return positions, ravel_fn


def SGHMCSVRG(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    centering_positions: Array,
    num_integration_steps: int,
    svrg_update_freq: int,
    rng_key: Array,
):
    num_cv_iterations = num_iterations / svrg_update_freq

    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sgldsvrg functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
    cv_full_logprob_grad = grad_fn(centering_positions, (X, y))

    schedule_fn = lambda _: step_size
    kern = kernels.sghmcsvrg(
        grad_fn, schedule_fn, (X, y), batches, svrg_update_freq, num_integration_steps
    )
    step_fn = kern.step

    # Get initial parameters and state
    _, rng_key = jax.random.split(rng_key)
    init_state = kern.init(
        init_positions, centering_positions, cv_full_logprob_grad, next(batches)
    )

    # Apply SGLD-SVRG with lax.scan
    def one_step(state, rng_key):
        last_state, all_states = step_fn(rng_key, state)
        return last_state, all_states.position

    _, rng_key = jax.random.split(rng_key)
    keys = jax.random.split(rng_key, num_cv_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, *x.shape[2:])), pytree
        )

    return positions, ravel_fn
