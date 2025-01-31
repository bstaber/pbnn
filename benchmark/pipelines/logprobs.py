import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.flatten_util import ravel_pytree


def logprior_fn(parameters):
    """Compute the value of the log-prior density function."""
    flat_params, _ = ravel_pytree(parameters)
    return jnp.sum(stats.norm.logpdf(flat_params))


def homoscedastic_loglike_fn(parameters, data, network, noise_level):
    """Gaussian homoscedastic log-likelihood with known noise level and for a given network."""
    X, y = data
    return -jnp.sum(
        0.5 * (y - network.apply({"params": parameters}, X)) ** 2 / noise_level**2
    )


def homoscedastic_dropout_loglike_fn(
    parameters,
    data,
    dropout_rng,
    network,
    noise_level,
):
    """Gaussian homoscedastic log-likelihood with known noise level and for a given network with dropout layers."""
    X, y = data
    return -jnp.sum(
        0.5
        * (y - network.apply({"params": parameters}, X, rngs={"dropout": dropout_rng}))
        ** 2
        / noise_level**2
    )


def heteroscedastic_loglike_fn(parameters, data, network):
    """Gaussian heteroscedastic log-likelihood for a given network."""
    X, y = data
    pred = network.apply({"params": parameters}, X)
    mu, log_sig2 = pred[0], pred[1]
    return -0.5 * log_sig2 - jnp.sum(0.5 * (y - mu) ** 2 / jnp.exp(log_sig2))


def heteroscedastic_dropout_loglike_fn(parameters, data, dropout_rng, network):
    """Gaussian heteroscedastic log-likelihood for a given network with dropout layers."""
    X, y = data
    pred = network.apply({"params": parameters}, X, rngs={"dropout": dropout_rng})
    mu, log_sig2 = pred[0], pred[1]
    return -0.5 * log_sig2 - jnp.sum(0.5 * (y - mu) ** 2 / jnp.exp(log_sig2))
