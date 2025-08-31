"""Module that contains the experiments used in the benchmark."""

from typing import Callable, NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import logprobs
import numpy as np
import torch
from jax import Array

from pbnn.models import MLP, MLPDropout
from pbnn.utils.analytical_functions import (
    barber_fn,
    gramacy_function,
    trigonometric_function,
)


class StandardScaler:
    """Standard Scaler class that scales the data to have zero mean and unit variance."""

    def __init__(self):
        """Initializes the StandardScaler."""
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Fits the scaler to the data."""
        self.mean_ = jnp.mean(X, axis=0)
        self.scale_ = jnp.std(X, axis=0)
        return self

    def transform(self, X):
        """Function that scales the data."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Function that fits and scales the data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """Function that inverts the scaling."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return X_scaled * self.scale_ + self.mean_


class DataSplit(NamedTuple):
    """NamedTuple that contains the information of a data split."""

    X_train: Array
    f_train: Array
    y_train: Array
    X_test: Array
    f_test: Array
    y_test: Array


LoadDataFn = Callable[[int], DataSplit]


class Experiment(NamedTuple):
    """NamedTuple that contains the information of an experiment."""

    name: str
    in_feats: int
    out_feats: int
    load_data_fn: LoadDataFn
    network: Callable[[], nn.Module]
    network_dropout: Callable[[], nn.Module]
    network_torch: Callable[[], torch.nn.Module]
    logprior_fn: Callable
    loglikelihood_fn: Callable
    loglikelihood_fn_dropout: Callable
    data_size: int
    noise_level: float
    scaler: StandardScaler


def make_experiment(
    name: str,
    data_fn: LoadDataFn,
    input_range: Tuple[float, float],
    dropout: float,
    in_feats: int,
    out_feats: int,
    data_size: int,
    noise_level: float,
    heteroscedastic: bool = False,
) -> Experiment:
    """Function that creates an experiment given the data function and the input range.

    Args:
        name: Name of the experiment
        data_fn: Data function
        input_range: Input range
        dropout: Dropout rate
        in_feats: Number of input features
        out_feats: Number of output features
        data_size: Size of the dataset
        noise_level: Level of noise
        heteroscedastic: Boolean that indicates if the noise is heteroscedastic

    Returns: Experiment for the given parameters.
    """
    in_scaler = StandardScaler()

    def load_data_fn(dataset_idx):
        np.random.seed(dataset_idx)
        X = np.random.uniform(
            low=input_range[0], high=input_range[1], size=(data_size, 1)
        )
        noise = noise_level * np.random.randn(data_size, 1)

        X, noise = jnp.array(X), jnp.array(noise)
        f, y = data_fn(X, noise)

        X_test = np.random.uniform(
            low=input_range[0], high=input_range[1], size=(data_size, 1)
        )
        noise_test = noise_level * np.random.randn(data_size, 1)
        X_test, noise_test = jnp.array(X_test), jnp.array(noise_test)
        f_test, y_test = data_fn(X_test, noise_test)

        X = in_scaler.fit_transform(X)
        X_test = in_scaler.transform(X_test)
        return X, f, y, X_test, f_test, y_test

    def network():
        return MLP(hidden_features=100, out_features=out_feats)

    def network_dropout():
        return MLPDropout(hidden_features=100, out_features=out_feats, dropout=dropout)

    def network_torch():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_feats),
        )

    if heteroscedastic:
        loglikelihood_fn = jax.tree_util.Partial(
            logprobs.heteroscedastic_loglike_fn, network=network()
        )
        loglikelihood_fn_dropout = jax.tree_util.Partial(
            logprobs.heteroscedastic_dropout_loglike_fn, network=network_dropout()
        )
    else:
        loglikelihood_fn = jax.tree_util.Partial(
            logprobs.homoscedastic_loglike_fn,
            network=network(),
            noise_level=noise_level,
        )
        loglikelihood_fn_dropout = jax.tree_util.Partial(
            logprobs.homoscedastic_dropout_loglike_fn,
            network=network_dropout(),
            noise_level=noise_level,
        )

    # loglikelihood_fn = jax.tree_util.Partial(
    #     logprobs.homoscedastic_loglike_fn, network=network(), noise_level=noise_level
    # )
    # loglikelihood_fn_dropout = jax.tree_util.Partial(
    #     logprobs.homoscedastic_dropout_loglike_fn, network=network_dropout(), noise_level=noise_level
    # )
    logprior_fn = logprobs.logprior_fn

    return Experiment(
        name,
        in_feats,
        out_feats,
        load_data_fn,
        network,
        network_dropout,
        network_torch,
        logprior_fn,
        loglikelihood_fn,
        loglikelihood_fn_dropout,
        data_size,
        noise_level,
        in_scaler,
    )


def Experiment1(dropout: float):
    """Experiment 1: Trigonometric function with additive homoscedastic noise."""
    return make_experiment(
        name="trigonometric_function",
        data_fn=trigonometric_function,
        input_range=(-3, 3),
        dropout=dropout,
        in_feats=1,
        out_feats=1,
        data_size=100,
        noise_level=0.04,
    )


def Experiment2(dropout: float):
    """Experiment 2: Gramacy function with additive homoscedastic noise."""
    return make_experiment(
        name="gramacy_function",
        data_fn=gramacy_function,
        input_range=(0, 20),
        dropout=dropout,
        in_feats=1,
        out_feats=1,
        data_size=100,
        noise_level=0.04,
    )


def Experiment3(dropout: float):
    """Experiment 3: Barber function with heteroscedastic noise."""
    return make_experiment(
        name="barber",
        data_fn=barber_fn,
        input_range=(0, 20),
        dropout=dropout,
        in_feats=1,
        out_feats=2,
        data_size=100,
        noise_level=0.04,
        heteroscedastic=True,
    )


def load_experiment(index: int, dropout: float = 0.0) -> Experiment:
    """Function that loads an experiment given the index.

    Args:
        index: Index of the experiment
        dropout: Dropout rate

    Returns: Experiment for the given index.
    """
    match index:
        case 1:
            return Experiment1(dropout)
        case 2:
            return Experiment2(dropout)
        case 3:
            return Experiment3(dropout)
        case _:
            raise ValueError()


if __name__ == "__main__":
    """Main function that loads an experiment and prints its information."""
    import rich

    # Load experiment 1
    experiment = load_experiment(index=3, dropout=0.0)

    # Print experiment information
    rich.print(experiment._asdict())
