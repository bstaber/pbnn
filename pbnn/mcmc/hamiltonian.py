from typing import Callable, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from blackjax.sgmcmc.gradients import grad_estimator
from jax import Array
from jax.flatten_util import ravel_pytree

from pbnn.mcmc import kernels
from pbnn.mcmc.sgmcmc.gradients import cv_grad_estimator
from pbnn.utils.data import batch_labeled_data as batch_data


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


def sghmc(
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
    """Wrapper of the SGHMC algorithm implemented in BlackJAX.

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
    num_integration_steps
        Number of leapfrog steps
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

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def scheduled_sghmc(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    schedule_fn: Callable[[int], float],
    num_iterations: int,
    num_integration_steps: int,
    rng_key: Array,
):
    """Wrapper of the cyclical SGHMC algorithm implemented in BlackJAX.

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
        Callable that gives the step size w.r.t the current iteration
    num_iterations
        Total number of iterations
    num_integration_steps
        Number of leapfrog steps
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

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def sghmc_cv(
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
    """SGHMC-CV algorithm implemented by relying on BlackJAX.

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
    num_integration_steps
        Number of leapfrog steps
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

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


def sghmc_svrg(
    X: Array,
    y: Array,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    init_positions: Array,
    batch_size: int,
    step_size: float,
    centering_positions: Array,
    num_cv_iterations: int,
    num_svrg_iterations: int,
    num_integration_steps: int,
    rng_key: Array,
):
    """SGHMC-SVRG algorithm implemented using BlackJAX.

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

    # svrg functions
    def one_svrg_step(state, rng_key):
        positions, centering_positions = state
        grad_fn = cv_grad_estimator(
            logprior_fn, loglikelihood_fn, (X, y), centering_positions
        )

        kern = blackjax.sghmc(grad_fn, num_integration_steps)
        sghmc_step_fn = kern.step

        # get initial state
        init_state = kern.init(positions, next(batches))

        @jax.jit
        def one_cv_step(state, rng_key):
            batch = next(batches)
            new_state = sghmc_step_fn(rng_key, state, batch, step_size)
            return new_state, new_state

        keys = jax.random.split(rng_key, num_cv_iterations)
        last_position, positions = jax.lax.scan(one_cv_step, init_state, keys)

        return (last_position, last_position), positions

    keys = jax.random.split(rng_key, num_svrg_iterations)
    _, positions = jax.lax.scan(
        one_svrg_step, (init_positions, centering_positions), keys
    )

    def reshape_fn(pytree):
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, *x.shape[2:])), pytree
        )

    positions = reshape_fn(positions)

    def ravel_fn(pytree):
        return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)

    def predict_fn(network, params, X_test):
        return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

    return positions, ravel_fn, predict_fn


# def sghmc_svrg(
#     X: Array,
#     y: Array,
#     loglikelihood_fn: Callable,
#     logprior_fn: Callable,
#     init_positions: Array,
#     batch_size: int,
#     step_size: float,
#     num_iterations: int,
#     centering_positions: Array,
#     num_integration_steps: int,
#     svrg_update_freq: int,
#     rng_key: Array,
# ):
#     """SGHMC-SVRG algorithm implemented by relying on BlackJAX.

#     Parameters
#     ----------

#     X
#         Matrix of input features of size (N, d)
#     y
#         Matrix of output features of size (N, s)
#     loglikelihood_fn
#         Callable log-likelihood function
#     logprior_fn
#         Callable log-prior function
#     init_positions
#         PyTree of initial positions
#     batch_size
#         Batch size for the stochastic gradient estimator
#     step_size
#         Step size
#     num_iterations
#         Total number of iterations
#     centering_positions
#         PyTree of control variates
#     num_integration_steps
#         Number of leapfrog steps
#     svrg_update_freq
#         Frequency at which the control state is updated
#     rng_key
#         Random seed key

#     Returns
#     -------

#     positions
#         Markov chain given as a PyTree
#     ravel_fn
#         Ravel function to flatten the PyTree
#     predict_fn
#         Function to make predictions

#     """
#     num_cv_iterations = num_iterations / svrg_update_freq

#     # batch data
#     data_size = len(X)
#     batches = batch_data(rng_key, (X, y), batch_size, data_size, replace=True)

#     # sgldsvrg functions
#     grad_fn = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
#     cv_full_logprob_grad = grad_fn(centering_positions, (X, y))

#     def schedule_fn(_):
#         return step_size

#     kern = kernels.sghmcsvrg(
#         grad_fn, schedule_fn, (X, y), batches, svrg_update_freq, num_integration_steps
#     )
#     step_fn = kern.step

#     # Get initial parameters and state
#     _, rng_key = jax.random.split(rng_key)
#     init_state = kern.init(
#         init_positions, centering_positions, cv_full_logprob_grad, next(batches)
#     )

#     # Apply SGLD-SVRG with lax.scan
#     def one_step(state, rng_key):
#         last_state, all_states = step_fn(rng_key, state)
#         return last_state, all_states.position

#     _, rng_key = jax.random.split(rng_key)
#     keys = jax.random.split(rng_key, num_cv_iterations)
#     _, positions = jax.lax.scan(one_step, init_state, keys)

#     def ravel_fn(pytree):
#         return jax.tree_util.tree_map(
#             lambda x: jnp.reshape(x, (-1, *x.shape[2:])), pytree
#         )

#     def predict_fn(network, params, X_test):
#         return jax.vmap(lambda p: network().apply({"params": p}, X_test), 0)(params)

#     return positions, ravel_fn, predict_fn
