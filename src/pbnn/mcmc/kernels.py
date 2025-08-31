"""Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD) kernel."""
# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

from typing import Callable

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, PRNGKey

import pbnn.mcmc.sgmcmc as sgmcmc

__all__ = [
    "psgld",
]


class psgld:
    """Implements the (basic) user interface for the preconditioned SGLD kernel.

    The general psgld kernel (:meth:`blackjax.mcmc.psgld.kernel`, alias `blackjax.psgld.kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    Example:
    -------
    To initialize a pSGLD kernel one needs to specify a schedule function, which
    returns a step size at each sampling step, and a gradient estimator
    function. Here for a constant step size, and `data_size` data samples:

    .. code::

        schedule_fn = lambda _: 1e-3
        grad_fn = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    Assuming we have an iterator `batches` that yields batches of data, we can initialize the psgld kernel and the state:

    .. code::

        psgld = blackjax.psgld(grad_fn, schedule_fn, alpha)
        state = psgld.init(position, next(batches))

    Note that this algorithm also takes an optional argument, alpha, which corresponds to the exponential decay
    of the preconditioner. When omiited, its value is set to 0.95.

    We can now perform one step that generates a new state:

    .. code::

        data_batch = next(batches)
        new_state = psgld.step(rng_key, state, data_batch)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(psgld.step)
       new_state, info = step(rng_key, state)

    Parameters
    ----------
    gradient_estimator_fn
       A function which, given a position and a batch of data, returns an estimation
       of the value of the gradient of the log-posterior distribution at this position.
    schedule_fn
       A function which returns a step size given a step number.
    alpha
       A float corresponding to the exponential decay of the preconditioning matrix.

    Returns:
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(sgmcmc.psgld.init)
    kernel = staticmethod(sgmcmc.psgld.kernel)

    def __new__(  # type: ignore[misc]
        cls,
        grad_estimator_fn: Callable,
        alpha: float = 0.95,
    ) -> SamplingAlgorithm:
        step = cls.kernel(grad_estimator_fn)

        def init_fn(position: Array, data_batch: Array):
            return cls.init(position, data_batch, grad_estimator_fn)

        def step_fn(rng_key: PRNGKey, state, data_batch: Array, step_size):
            return step(rng_key, state, data_batch, step_size, alpha)

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
