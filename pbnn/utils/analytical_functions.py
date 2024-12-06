# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from flax import linen as nn


def trigonometric_function(x: Array, noise: Array) -> Array:
    r"""Simple trigonometric function of the form:

    .. math::

        y(x) = \cos(2*x) + sin(x) + \epsilon(x)

    where :math:`\epsilon` is the additive noise.

    Parameters
    ~~~~~~~~~~
    x
        Input array of size (n, 1)
    noise
        Noise array of size (n, 1)


    Example
    ~~~~~~~

    Homoscedastic regression:

    .. code-block::

        import jax.random as jr
        keys = jr.splt(jr.PRNGKey(0), 2)

        x = jr.uniform(keys[0], minval=-3, maxval=3, shape=(100, 1))
        noise = jr.normal(keys[1], shape=(100, 1))

        y = trigonometric_function1(x, noise)

    Heteroscedastic regression:

    .. code-block::

        import jax.random as jr
        keys = jr.splt(jr.PRNGKey(0), 2)

        x = jr.uniform(keys[0], minval=-3, maxval=3, shape=(100, 1))
        noise = jr.normal(keys[1], shape=(100, 1)) * jnp.abs(x)**2

        y = trigonometric_function1(x, noise)

    Returns
    ~~~~~~~
    y
        Values of the function evaluated at the given inputs
    """
    f = jnp.cos(2.0 * x) + jnp.sin(x)
    y = f + noise
    return f, y

def heteroscedastic_trigonometric_function(x: Array, noise: Array) -> Array:
    r"""Simple trigonometric function of the form:

    .. math::

        y(x) = 2*\sin(2*\beta^T x) + \pi*\beta^2 x + \sqrt{1.0 + (\beta^T x)^2}

    where :math:`\epsilon` is a Gaussian random variable.

    Parameters
    ~~~~~~~~~~
    x
        Input array of size (n, d)
    noise
        Noise array of size (n, 1)


    Example
    ~~~~~~~

    Homoscedastic regression:

    .. code-block::

        import jax.random as jr
        keys = jr.splt(jr.PRNGKey(0), 2)

        x = jr.uniform(keys[0], minval=-3, maxval=3, shape=(100, 1))
        noise = jr.normal(keys[1], shape=(100, 1))

        y = trigonometric_function2(x, noise)

    Heteroscedastic regression:

    .. code-block::

        import jax.random as jr
        keys = jr.splt(jr.PRNGKey(0), 2)

        x = jr.uniform(keys[0], minval=-3, maxval=3, shape=(100, 1))
        noise = jr.normal(keys[1], shape=(100, 1)) * jnp.abs(x)**2

        y = trigonometric_function2(x, noise)

    Returns
    ~~~~~~~
    y
        Values of the function evaluated at the given inputs
    """

    dim = x.shape[-1]
    beta = jnp.zeros((dim,))
    beta = beta.at[0:5].set(1.0)
    z = jnp.dot(x, beta)[:, None]
    y = 2.0*jnp.sin(jnp.pi*z) + jnp.pi*z + jnp.sqrt(1.0 + z**2) * noise
    return y


def mlp_function(x: Array, params_rng_key: Array, noise: Array) -> Array:
    r"""Regression function generated from a MLP network
    with parameters drawn from a normal distribution.

    Given an input :math:`x`, we first build the features :math:`z_1 = x_1`
    and :math:`z_2`, and the output computed as follows:

    .. math::

        y(z) = \mathrm{MLP}(z; \mathbf{\theta}_0) + \epsilon(z)

    where :math:`\mathbf{\theta}_0` are fixed parameters drawn from a normal distribution,
    and :math:`\epsilon` is the addive noise specified by the user.

    Parameters
    ~~~~~~~~~~
    x
        Input array of size (n, 1)
    params_rng_key
        PRNGKey for the normally distributed MLP parameters
    noise
        Noise array of size (n, 1)

    Returns
    ~~~~~~~
    y
        Values of the function evaluated at the given inputs
    """

    class MLP_relu(nn.Module):
        """Simple MLP."""

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(
                features=100,
                kernel_init=nn.initializers.normal(),
                bias_init=nn.initializers.normal(),
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                features=100,
                kernel_init=nn.initializers.normal(),
                bias_init=nn.initializers.normal(),
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                features=100,
                kernel_init=nn.initializers.normal(),
                bias_init=nn.initializers.normal(),
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                features=1,
                kernel_init=nn.initializers.normal(),
                bias_init=nn.initializers.normal(),
            )(x)
            return x

    # generate dummy params
    keys = jr.split(params_rng_key, 2)
    params = MLP_relu().init(keys[0], jnp.ones(100, 2))["params"]

    # ravel to get the unravel_fn
    flat_params, unravel_fn = ravel_pytree(params)

    # generate gaussian params and unravel them
    flat_random_params = 0.1 * jr.normal(keys[1], shape=flat_params.shape)
    random_params = unravel_fn(flat_random_params)

    # generate data
    features = jnp.concatenate([x, x**2], axis=1)
    y = MLP_relu().apply({"params": random_params}, features) + noise
    return y


def ishigami_function(x: Array, noise: Array, a: int = 7, b: int = 0.1) -> Array:
    r"""Ishigami function defined as:

    .. math::

        y(x) = \sin(x_1) + a\sin^2(x_2) + bx_3^4\sin(x_1) + \epsilon(x)

    where :math:`\epsilon` is the additive noise.

    Parameters
    ~~~~~~~~~~
    x
        Input array of size (n, 3)
    noise
        Noise array of size (n, 1)
    a
        Scaling parameter (default value: 7)
    b
        Scaling parameter (default value: 0.1)

    Returns
    ~~~~~~~
    y
        Values of the function evaluated at the given inputs
    """

    return (
        jnp.sin(x[:, 0])
        + a * jnp.sin(x[:, 1]) ** 2
        + b * x[:, 2] ** 4 * jnp.sin(x[:, 0])
        + noise
    )


def gramacy_function(x: Array, noise: Array) -> Array:
    r"""Non-stationnary test function from Gramacy (2007).

    Parameters
    ~~~~~~~~~~
    x
        Input array of size (n, 1)
    noise
        Noise array of size (n, 1)

    Returns
    ~~~~~~~
    y
        Values of the function evaluated at the given inputs
    """

    idx_1 = x[:, 0] <= 9.6
    idx_2 = x[:, 0] > 9.6
    f = jnp.zeros_like(x)
    f = f.at[idx_1].set(
        jnp.sin(jnp.pi * x[idx_1] / 5.0)
        + (1.0 / 5.0) * jnp.cos(4.0 * jnp.pi * x[idx_1] / 5.0)
    )
    f = f.at[idx_2].set(x[idx_2] / 10.0 - 1.0)
    y = f + noise
    return f, y


def g_function(x: Array, noise: Array) -> Array:
    r"""G-function defined as:

    .. math::

        f(\mathbf{x}) = \prod_{i=1}^{d} \frac{|4x_i - 2| + a_i}{1 + a_i}\,,

    where

    .. math::

        a_i = \frac{i-2}{2}\,, \quad i = 1, \dots, d\,.

    Parameters
    ~~~~~~~~~~
    x
        Input array of size (n, 1)
    noise
        Noise array of size (n, 1)

    Returns
    ~~~~~~~
    y
        Values of the function evaluated at the given inputs
    """
    d = x.shape[1]
    a = (jnp.arange(d) + 1 - 2.0) / 2.0
    f = jnp.prod((jnp.abs(4.0 * x - 1.0) + a[None]) / (1.0 + a[None]), axis=1)
    y = f + noise
    return y
