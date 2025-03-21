# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

import jax.numpy as jnp
import jax.scipy.stats as stats
from flax import linen as nn
from jax.flatten_util import ravel_pytree


class MLP(nn.Module):
    """Simple MLP."""

    hidden_features: int
    out_features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=self.hidden_features,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.hidden_features,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.out_features,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
        )(x)
        return x


class MLPDropout(nn.Module):
    """Simple MLP with dropout."""

    hidden_features: int
    out_features: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = nn.Dense(
            features=self.hidden_features,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
        )(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.hidden_features,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
        )(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.out_features,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
        )(x)
        return x
