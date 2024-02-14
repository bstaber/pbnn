from typing import Callable
import jax
from blackjax.types import Array
from blackjax.sgmcmc.gradients import grad_estimator


def cv_grad_estimator(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    data: Array,
    centering_position: Array,
) -> Callable:
    """Builds a control variate gradient estimator [1]_.
    Parameters
    ----------
    logprior_fn
        The log-probability density function corresponding to the prior
        distribution.
    loglikelihood_fn
        The log-probability density function corresponding to the likelihood.
    data
        The full dataset.
    centering_position
        Centering position for the control variates (typically the MAP).
    References
    ----------
    .. [1]: Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2019).
            Control variates for stochastic gradient MCMC. Statistics
            and Computing, 29(3), 599-615.
    """
    data_size = jax.tree_util.tree_leaves(data)[0].shape[0]
    logposterior_grad_estimator_fn = grad_estimator(
        logprior_fn, loglikelihood_fn, data_size
    )

    # Control Variates use the gradient on the full dataset
    logposterior_grad_center = logposterior_grad_estimator_fn(centering_position, data)

    def logposterior_estimator_fn(position: Array, data_batch: Array) -> float:
        """Return an approximation of the log-posterior density.
        Parameters
        ----------
        position
            The current value of the random variables.
        batch
            The current batch of data. The first dimension is assumed to be the
            batch dimension.
        Returns
        -------
        An approximation of the value of the log-posterior density function for
        the current value of the random variables.
        """
        logposterior_grad_estimate = logposterior_grad_estimator_fn(
            position, data_batch
        )
        logposterior_grad_center_estimate = logposterior_grad_estimator_fn(
            centering_position, data_batch
        )

        def control_variate(grad_estimate, center_grad_estimate, center_grad):
            return grad_estimate + center_grad - center_grad_estimate

        return jax.tree_util.tree_map(
            control_variate,
            logposterior_grad_estimate,
            logposterior_grad_center_estimate,
            logposterior_grad_center,
        )

    return logposterior_estimator_fn
