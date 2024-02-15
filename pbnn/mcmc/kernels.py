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

    Example
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

    Returns
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


# class ScheduleState(NamedTuple):
#     """State of the scheduleing."""

#     step_size: float
#     do_sample: bool


# class CyclicalSGMCMCState(NamedTuple):
#     """State of the Cyclical SGMCMC sampler."""

#     position: Array
#     opt_state: optax.OptState


# class CyclicalSGLD:
#     """Adapted from: https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html"""

#     sgd = optax.sgd(1)

#     def __new__(
#         cls,
#         grad_estimator_fn: Callable,
#         num_training_steps: int,
#         initial_step_size: float,
#         num_cycles: int = 4,
#         exploration_ratio: float = 0.25,
#     ) -> SamplingAlgorithm:
#         sgld = blackjax.sgld(grad_estimator_fn)

#         cycle_length = num_training_steps // num_cycles

#         def schedule_fn(step_id):
#             do_sample = jax.lax.cond(
#                 ((step_id % cycle_length) / cycle_length) >= exploration_ratio,
#                 lambda _: True,
#                 lambda _: False,
#                 step_id,
#             )

#             cos_out = jnp.cos(jnp.pi * (step_id % cycle_length) / cycle_length) + 1
#             step_size = 0.5 * cos_out * initial_step_size

#             return ScheduleState(step_size, do_sample)

#         def init_fn(position: Array):
#             opt_state = cls.sgd.init(position)
#             return CyclicalSGMCMCState(position, opt_state), schedule_fn

#         def step_fn(rng_key, state, minibatch, schedule_state):
#             """Cyclical SGD-SGLD kernel."""

#             def step_with_sgld(current_state):
#                 rng_key, state, minibatch, step_size = current_state
#                 new_position = sgld.step(rng_key, state.position, minibatch, step_size)
#                 return new_position, state.opt_state

#             def step_with_sgd(current_state):
#                 _, state, minibatch, step_size = current_state
#                 grads = grad_estimator_fn(state.position, minibatch)
#                 rescaled_grads = jax.tree_util.tree_map(
#                     lambda g: -1.0 * step_size * g, grads
#                 )
#                 updates, new_opt_state = cls.sgd.update(
#                     rescaled_grads, state.opt_state, state.position
#                 )
#                 new_position = optax.apply_updates(state.position, updates)
#                 return new_position, new_opt_state

#             new_position, new_opt_state = jax.lax.cond(
#                 schedule_state.do_sample,
#                 step_with_sgld,
#                 step_with_sgd,
#                 (rng_key, state, minibatch, schedule_state.step_size),
#             )

#             return CyclicalSGMCMCState(new_position, new_opt_state)

#         return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# class sgldsvrg:
#     """Implements the (basic) user interface for the SGLD-SVRG kernel.

#     The general sgldsvrg kernel (:meth:`blackjax.mcmc.sgldsvrg.kernel`, alias `blackjax.sgldsvrg.kernel`) can be
#     cumbersome to manipulate. Since most users only need to specify the kernel
#     parameters at initialization time, we provide a helper function that
#     specializes the general kernel.

#     This interface is based on the sgldcv interface. The SVRG algorithm consists in
#     updating the control state of the SGLDCV kernel with a constant frequency
#     (e.g., every 100 iterations).

#     Example
#     -------

#     To initialize a SGLD-SVRG kernel one needs to specify a schedule function, which
#     returns a step size at each sampling step, a gradient estimator
#     function, a training dataset, a batch generator, and the update frequency.
#     Here for a constant step size, and `data_size` data samples:

#     .. code::

#         schedule_fn = lambda _: 1e-3
#         grad_fn = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

#     We can now initialize the sgldsvrg kernel as follows:

#     .. code::

#         sgldsvrg = blackjax.sgldsvrg(grad_fn, schedule_fn, train_dataset, batch_loader, update_freq)

#     The state can be initialized by specifying an initial position, the control position that
#     corresponds to an estimate of the MAP, the gradient of the log-target computed with the
#     full dataset at the control position. Assuming we have an iterator `batches` that yields
#     batches of data, the sgldcv kernel is initialized as follows:

#     .. code::

#         init_batch = next(batch_loader)
#         state = sgldsvrg.init(position, map_position, cv_full_logprob_grad, init_batch)

#     One step of the sgldsvrg kernel performs `update_freq` sgldcv steps and updates the
#     control state with the latest sgld state. One step can be performed as follows:

#     .. code::

#         new_state, sgldcv_states = sgldsvrg.step(rng_key, state)

#     Here, `last_state` corresponds to the last sgldcv state with an updated control state,
#     and `sgldcv_states` corresponds to the `updated_freq` concatenated sgldcv states.

#     Kernels are not jit-compiled by default so you will need to do it manually:

#     .. code::

#        step = jax.jit(sgldcv.step)
#        new_state, sgldcv_states = step(rng_key, state)

#     Parameters
#     ----------
#     gradient_estimator_fn
#        A function which, given a position and a batch of data, returns an estimation
#        of the value of the gradient of the log-posterior distribution at this position.
#     schedule_fn
#        A function which returns a step size given a step number.

#     Returns
#     -------
#     A ``SamplingAlgorithm``.

#     """

#     init = staticmethod(sgmcmc.sgldcv.init)
#     kernel = staticmethod(sgmcmc.sgldcv.kernel)

#     def __new__(  # type: ignore[misc]
#         cls,
#         grad_estimator_fn: Callable,
#         schedule_fn: Callable,
#         train_dataset,
#         batch_loader,
#         update_freq: int,
#     ) -> SamplingAlgorithm:
#         step = cls.kernel(grad_estimator_fn)

#         def init_fn(
#             position: Array,
#             c_position: Array,
#             c_full_loglike_grad: Array,
#             data_batch: Array,
#         ):
#             return cls.init(
#                 position, c_position, c_full_loglike_grad, data_batch, grad_estimator_fn
#             )

#         def step_fn(rng_key: PRNGKey, state):
#             step_size = schedule_fn(state.step)

#             def svrg_kernel_step(state, rng_key):
#                 batch = next(batch_loader)
#                 new_state = step(rng_key, state, batch, step_size)
#                 return new_state, new_state

#             keys = jax.random.split(rng_key, update_freq)
#             last_svrg_state, svrg_states = jax.lax.scan(svrg_kernel_step, state, keys)
#             c_full_logprob_grad = grad_estimator_fn(
#                 last_svrg_state.position, train_dataset
#             )
#             updated_state = sgmcmc.sgldcv.SGLDCVState(
#                 last_svrg_state.step,
#                 last_svrg_state.position,
#                 last_svrg_state.batch_logprob_grad,
#                 last_svrg_state.position,
#                 c_full_logprob_grad,
#                 last_svrg_state.batch_logprob_grad,
#             )
#             return updated_state, svrg_states

#         return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# class sghmcsvrg:
#     """Implements the (basic) user interface for the SGHMC-SVRG kernel.

#     The general sghmcsvrg kernel (:meth:`blackjax.mcmc.sghmcsvrg.kernel`, alias `blackjax.sghmcsvrg.kernel`) can be
#     cumbersome to manipulate. Since most users only need to specify the kernel
#     parameters at initialization time, we provide a helper function that
#     specializes the general kernel.

#     This interface is based on the sghmccv interface. The SVRG algorithm consists in
#     updating the control state of the SGHMCCV kernel with a constant frequency
#     (e.g., every 100 iterations).

#     Example
#     -------

#     To initialize a SGHMC-SVRG kernel one needs to specify a schedule function, which
#     returns a step size at each sampling step, a gradient estimator
#     function, a training dataset, a batch generator, and the update frequency.
#     Here for a constant step size, and `data_size` data samples:

#     .. code::

#         schedule_fn = lambda _: 1e-3
#         grad_fn = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

#     We can now initialize the sghmcsvrg kernel as follows:

#     .. code::

#         sghmcsvrg = blackjax.sghmcsvrg(grad_fn, schedule_fn, train_dataset, batch_loader, update_freq)

#     The state can be initialized by specifying an initial position, the control position that
#     corresponds to an estimate of the MAP, the gradient of the log-target computed with the
#     full dataset at the control position. Assuming we have an iterator `batches` that yields
#     batches of data, the sghmccv kernel is initialized as follows:

#     .. code::

#         init_batch = next(batch_loader)
#         state = sghmcsvrg.init(position, map_position, cv_full_logprob_grad, init_batch)

#     One step of the sghmcsvrg kernel performs `update_freq` sghmccv steps and updates the
#     control state with the latest sghmc state. One step can be performed as follows:

#     .. code::

#         new_state, sghmccv_states = sghmcsvrg.step(rng_key, state)

#     Here, `last_state` corresponds to the last sghmccv state with an updated control state,
#     and `sghmccv_states` corresponds to the `updated_freq` concatenated sghmccv states.

#     Kernels are not jit-compiled by default so you will need to do it manually:

#     .. code::

#        step = jax.jit(sghmccv.step)
#        new_state, sghmccv_states = step(rng_key, state)

#     Parameters
#     ----------
#     gradient_estimator_fn
#        A function which, given a position and a batch of data, returns an estimation
#        of the value of the gradient of the log-posterior distribution at this position.
#     schedule_fn
#        A function which returns a step size given a step number.

#     Returns
#     -------
#     A ``SamplingAlgorithm``.

#     """

#     init = staticmethod(sgmcmc.sghmccv.init)
#     kernel = staticmethod(sgmcmc.sghmccv.kernel)

#     def __new__(  # type: ignore[misc]
#         cls,
#         grad_estimator_fn: Callable,
#         schedule_fn: Callable,
#         train_dataset,
#         batch_loader,
#         update_freq: int,
#         num_integration_steps: int,
#     ) -> SamplingAlgorithm:
#         step = cls.kernel(grad_estimator_fn)

#         def init_fn(
#             position: Array,
#             c_position: Array,
#             c_full_loglike_grad: Array,
#             data_batch: Array,
#         ):
#             return cls.init(
#                 position, c_position, c_full_loglike_grad, data_batch, grad_estimator_fn
#             )

#         def step_fn(rng_key: PRNGKey, state):
#             step_size = schedule_fn(state.step)

#             def svrg_kernel_step(state, rng_key):
#                 batch = next(batch_loader)
#                 new_state = step(
#                     rng_key, state, batch, step_size, num_integration_steps
#                 )
#                 return new_state, new_state

#             keys = jax.random.split(rng_key, update_freq)
#             last_svrg_state, svrg_states = jax.lax.scan(svrg_kernel_step, state, keys)
#             c_full_logprob_grad = grad_estimator_fn(
#                 last_svrg_state.position, train_dataset
#             )
#             updated_state = sgmcmc.sghmccv.SGLDCVState(
#                 last_svrg_state.step,
#                 last_svrg_state.position,
#                 last_svrg_state.batch_logprob_grad,
#                 last_svrg_state.position,
#                 c_full_logprob_grad,
#                 last_svrg_state.batch_logprob_grad,
#             )
#             return updated_state, svrg_states

#         return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
