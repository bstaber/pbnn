from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
import kernels
from blackjax.gradient import grad_estimator
from jax import Array
from jax.flatten_util import ravel_pytree

from pbnn.mcmc.sgmcmc.gradients import cv_grad_estimator
from pbnn.utils.data import batch_data


def sgld(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    rng_key: Array,
):
    """Wrapper of the SGLD algorithm implemented in BlackJAX.

    Parameters
    ----------

    X
        Matrix of input features of size (N, d)
    y
        Matrix of output features of size (N, s)
    loglikelihood_fn
        Callable log-likelihood function
    logprior_fn
        Callable log-prior function
    init_positions
        PyTree of initial positions
    batch_size
        Batch size for the stochastic gradient estimator
    step_size
        Step size
    num_iterations
        Total number of iterations
    rng_key
        Random seed key

    Returns
    -------

    positions
        Markov chain given as a PyTree
    ravel_fn
        Ravel function to flatten the PyTree
    predict_fn
        Function to make predictions

    """
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sgld functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
    kern = blackjax.sgld(grad_fn)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGLD with lax.scan
    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        new_state = step_fn(rng_key, state, batch, step_size)
        return new_state, new_state

    keys = jax.random.split(rng_key, num_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def pSGLD(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    preconditioning_factor: float,
    rng_key: Array,
):
    """Wrapper of the SGLD algorithm implemented in BlackJAX.

    Parameters
    ----------

    X
        Matrix of input features of size (N, d)
    y
        Matrix of output features of size (N, s)
    loglikelihood_fn
        Callable log-likelihood function
    logprior_fn
        Callable log-prior function
    init_positions
        PyTree of initial positions
    batch_size
        Batch size for the stochastic gradient estimator
    step_size
        Step size
    num_iterations
        Total number of iterations
    preconditiong_factor
        Preconditioning factor
    rng_key
        Random seed key

    Returns
    -------

    positions
        Markov chain given as a PyTree
    ravel_fn
        Ravel function to flatten the PyTree
    predict_fn
        Function to make predictions

    """
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sgld functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
    kern = kernels.psgld(grad_fn, preconditioning_factor)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGLD with lax.scan
    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        new_state = step_fn(rng_key, state, batch, step_size)
        return new_state, new_state

    keys = jax.random.split(rng_key, num_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def adaptive_sgld(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    schedule_fn: Callable[[int], float],
    num_iterations: int,
    rng_key: Array,
):
    """Wrapper of the SGLD algorithm implemented in BlackJAX.

    Parameters
    ----------

    X
        Matrix of input features of size (N, d)
    y
        Matrix of output features of size (N, s)
    loglikelihood_fn
        Callable log-likelihood function
    logprior_fn
        Callable log-prior function
    init_positions
        PyTree of initial positions
    batch_size
        Batch size for the stochastic gradient estimator
    schedule_fn
        Callable function that returns the step size w.r.t the current iteration
    num_iterations
        Total number of iterations
    rng_key
        Random seed key

    Returns
    -------

    positions
        Markov chain given as a PyTree
    ravel_fn
        Ravel function to flatten the PyTree
    predict_fn
        Function to make predictions

    """
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sgld functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
    kern = blackjax.sgld(grad_fn)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGLD with lax.scan
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

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def sgld_cv(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    centering_positions: Array,
    rng_key: Array,
):
    """Wrapper of the SGLD algorithm implemented in BlackJAX.

    Parameters
    ----------

    X
        Matrix of input features of size (N, d)
    y
        Matrix of output features of size (N, s)
    loglikelihood_fn
        Callable log-likelihood function
    logprior_fn
        Callable log-prior function
    init_positions
        PyTree of initial positions
    batch_size
        Batch size for the stochastic gradient estimator
    step_size
        Step size
    num_iterations
        Total number of iterations
    centering_positions
        PyTree of control variates
    rng_key
        Random seed key

    Returns
    -------

    positions
        Markov chain given as a PyTree
    ravel_fn
        Ravel function to flatten the PyTree
    predict_fn
        Function to make predictions

    """
    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sgld functions
    grad_fn = cv_grad_estimator(
        logprior_fn, loglikelihood_fn, (X, y), centering_positions
    )
    kern = blackjax.sgld(grad_fn)
    step_fn = jax.jit(kern.step)

    # get initial state
    init_state = kern.init(init_positions, next(batches))

    # apply SGLD with lax.scan
    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        new_state = step_fn(rng_key, state, batch, step_size)
        return new_state, new_state

    keys = jax.random.split(rng_key, num_iterations)
    _, positions = jax.lax.scan(one_step, init_state, keys)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def sgld_svrg(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    num_iterations: int,
    centering_positions: Array,
    svrg_update_freq: int,
    rng_key: Array,
):
    """SGHMC-SVRG algorithm implemented by relying on BlackJAX.

    Parameters
    ----------

    X
        Matrix of input features of size (N, d)
    y
        Matrix of output features of size (N, s)
    loglikelihood_fn
        Callable log-likelihood function
    logprior_fn
        Callable log-prior function
    init_positions
        PyTree of initial positions
    batch_size
        Batch size for the stochastic gradient estimator
    step_size
        Step size
    num_iterations
        Total number of iterations
    centering_positions
        PyTree of control variates
    svrg_update_freq
        Frequency at which the control state is updated
    rng_key
        Random seed key

    Returns
    -------

    positions
        Markov chain given as a PyTree
    ravel_fn
        Ravel function to flatten the PyTree
    predict_fn
        Function to make predictions

    """
    num_cv_iterations = num_iterations / svrg_update_freq

    # batch data
    data_size = len(X)
    batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

    # sgldsvrg functions
    grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
    cv_full_logprob_grad = grad_fn(centering_positions, (X, y))

    def schedule_fn(_):
        return step_size
    kern = kernels.sgldsvrg(grad_fn, schedule_fn, (X, y), batches, svrg_update_freq)
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

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn
