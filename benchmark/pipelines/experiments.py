from typing import Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import logprobs
import numpy as np
import torch
from pbnn.models import MLP, MLPDropout
from pbnn.utils.analytical_functions import gramacy_function, trigonometric_function


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = jnp.mean(X, axis=0)
        self.scale_ = jnp.std(X, axis=0)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return X_scaled * self.scale_ + self.mean_


class Experiment(NamedTuple):
    name: str
    in_feats: int
    out_feats: int
    load_data_fn: Callable
    network: nn.Module
    network_dropout: nn.Module
    network_torch: torch.nn.Module
    logprior_fn: Callable
    loglikelihood_fn: Callable
    loglikelihood_fn_dropout: Callable
    data_size: int
    noise_level: float
    scaler: StandardScaler


def Experiment1(dropout: float):
    name = "trigonometric_function"
    in_feats = 1
    out_feats = 1
    data_size = 100
    noise_level = 0.04
    in_scaler = StandardScaler()

    def load_data_fn(dataset_idx):
        np.random.seed(dataset_idx)
        X = np.random.uniform(low=-3, high=3, size=(data_size, 1))
        noise = noise_level * np.random.randn(data_size, 1)

        X, noise = jnp.array(X), jnp.array(noise)
        f, y = trigonometric_function(X, noise)

        X_test = np.random.uniform(low=-3, high=3, size=(data_size, 1))
        noise_test = noise_level * np.random.randn(data_size, 1)

        X_test, noise_test = jnp.array(X_test), jnp.array(noise_test)
        f_test, y_test = trigonometric_function(X_test, noise_test)

        X = in_scaler.fit_transform(X)
        X_test = in_scaler.transform(X_test)
        return X, f, y, X_test, f_test, y_test

    def network():
        return MLP(hidden_features=100, out_features=1)

    def network_dropout():
        return MLPDropout(hidden_features=100, out_features=1, dropout=dropout)

    def network_torch():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )

    loglikelihood_fn = jax.tree_util.Partial(
        logprobs.homoscedastic_loglike_fn, network=network(), noise_level=noise_level
    )
    loglikelihood_fn_dropout = jax.tree_util.Partial(
        logprobs.homoscedastic_dropout_loglike_fn,
        network=network_dropout(),
        noise_level=noise_level,
    )
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


def Experiment2(dropout: float):
    name = "gramacy_function"
    in_feats = 1
    out_feats = 1
    data_size = 100
    noise_level = 0.04
    in_scaler = StandardScaler()

    def load_data_fn(dataset_idx):
        np.random.seed(dataset_idx)
        X = np.random.uniform(low=0, high=20, size=(data_size, 1))
        noise = noise_level * np.random.randn(data_size, 1)

        X, noise = jnp.array(X), jnp.array(noise)
        f, y = gramacy_function(X, noise)

        X_test = np.random.uniform(low=0, high=20, size=(data_size, 1))
        noise_test = noise_level * np.random.randn(data_size, 1)

        X_test, noise_test = jnp.array(X_test), jnp.array(noise_test)
        f_test, y_test = gramacy_function(X_test, noise_test)

        X = in_scaler.fit_transform(X)
        X_test = in_scaler.transform(X_test)
        return X, f, y, X_test, f_test, y_test

    def network():
        return MLP(hidden_features=100, out_features=1)

    def network_dropout():
        return MLPDropout(hidden_features=100, out_features=1, dropout=dropout)

    def network_torch():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )

    loglikelihood_fn = jax.tree_util.Partial(
        logprobs.homoscedastic_loglike_fn, network=network(), noise_level=noise_level
    )
    loglikelihood_fn_dropout = jax.tree_util.Partial(
        logprobs.homoscedastic_dropout_loglike_fn,
        network=network_dropout(),
        noise_level=noise_level,
    )
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


def Experiment3(dropout: float):
    name = "barber"
    in_feats = 1
    out_feats = 2
    data_size = 200
    noise_level = 1.0
    in_scaler = StandardScaler()

    def load_data_fn(dataset_idx):
        np.random.seed(dataset_idx)
        X = np.random.uniform(low=0, high=20, size=(data_size, 1))
        noise = noise_level * np.random.randn(data_size, 1)

        X, noise = jnp.array(X), jnp.array(noise)
        f, y = X / 2.0, X / 2.0 + noise * jnp.abs(jnp.sin(X))

        X_test = np.random.uniform(low=0, high=20, size=(data_size, 1))
        noise_test = noise_level * np.random.randn(data_size, 1)

        X_test, noise_test = jnp.array(X_test), jnp.array(noise_test)
        f_test, y_test = (
            X_test / 2.0,
            X_test / 2.0 + noise_test * jnp.abs(jnp.sin(X_test)),
        )

        X = in_scaler.fit_transform(X)
        X_test = in_scaler.transform(X_test)
        return X, f, y, X_test, f_test, y_test

    def network():
        return MLP(hidden_features=50, out_features=2)

    def network_dropout():
        return MLPDropout(hidden_features=50, out_features=2, dropout=dropout)

    def network_torch():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2),
        )

    loglikelihood_fn = jax.tree_util.Partial(
        logprobs.heteroscedastic_loglike_fn, network=network(),
    )
    loglikelihood_fn_dropout = jax.tree_util.Partial(
        logprobs.heteroscedastic_dropout_loglike_fn,
        network=network_dropout(),
    )
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


def load_experiment(index: int, dropout: float = 0.0) -> Experiment:
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
    experiment = load_experiment(index=1, dropout=0.0)

    print(experiment)
