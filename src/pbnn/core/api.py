# src/pbnn/core/api.py
"""Core API definitions for probabilistic models and inference methods."""

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from jax import Array

__all__ = ["SupervisedBatch", "Posterior", "InferenceMethod"]


@dataclass
class SupervisedBatch:
    """A batch of supervised data.

    Attributes:
        x (Array): Input features with leading batch dimension,
            e.g., shape ``(B, d_in, ...)``.
        y (Array): Targets compatible with ``x``, e.g., shape ``(B, d_out, ...)``.
    """

    x: Array
    y: Array


@runtime_checkable
class Posterior(Protocol):
    """Posterior distribution over model predictions or parameters."""

    def predictive_mean_var(self, x: Array, **kwargs) -> Tuple[Array, Array]:
        """Compute predictive mean and variance at inputs.

        Args:
            x (Array): Inputs with leading batch dimension.
            **kwargs: Method-specific keyword arguments
                (e.g., ``train=False``, RNGs, Flax `mutable`).

        Returns:
            Tuple[Array, Array]: A tuple ``(mean, var)`` where
            - ``mean`` has the same shape as the model outputs for ``x``.
            - ``var`` is the elementwise predictive variance
              (zero for point estimates such as MAP).
        """
        ...

    def predict(self, x: Array, **kwargs) -> Array:
        """Deterministic prediction (typically the predictive mean).

        Args:
            x (Array): Inputs with leading batch dimension.
            **kwargs: Method-specific keyword arguments.

        Returns:
            Array: Predicted mean values with the same shape as
            ``predictive_mean_var(x)[0]``.
        """
        ...


@runtime_checkable
class InferenceMethod(Protocol):
    """Interface for training probabilistic models.

    Implementations consume a model and supervised data, and return
    a `Posterior` object that exposes a method-agnostic predictive API.
    """

    def fit(
        self,
        model: Any,
        train_ds: Dict[str, Array],
        valid_ds: Dict[str, Array] | None = None,
        **kwargs,
    ) -> Posterior:
        """Train on supervised data and return a posterior.

        Args:
            model (Any): Model or architecture to be trained
                (e.g., a Flax `nn.Module`).
            train_ds (Dict[str, Array]): Training dataset with keys:
                - ``"x"``: inputs (Array with leading batch dimension).
                - ``"y"``: targets (Array compatible with ``"x"``).
            valid_ds (Dict[str, Array] | None, optional): Validation dataset
                in the same format as ``train_ds``. Defaults to None.
            **kwargs: Additional method-specific parameters (optimizers,
                schedules, seeds, etc.).

        Returns:
            Posterior: Posterior object exposing
            ``predict`` and ``predictive_mean_var``.

        Notes:
            Implementations should avoid side effects outside the returned
            posterior object (e.g., no global state).
        """
        ...
