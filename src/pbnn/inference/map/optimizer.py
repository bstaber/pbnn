"""MAP estimation with JAX/Flax: fast, minimal, and API-friendly."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import Array

from pbnn.utils.data import NumpyDataset, NumpyLoader


@dataclass
class MAPConfig:
    """Configuration for MAP training."""

    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam" | "sgd"
    batch_size: int = 128
    num_epochs: int = 100
    clip_grad_norm: Optional[float] = None
    weight_decay: float = 0.0  # decoupled (AdamW-style) if > 0
    seed: int = 0


class PosteriorMAP:
    """Posterior wrapper for MAP: point estimate + module apply."""

    def __init__(self, params, apply_fn: Callable):
        self.params = params
        self._apply = apply_fn

    def predict(self, x: Array, **apply_kwargs) -> Array:
        """Deterministic prediction (mean)."""
        return self._apply({"params": self.params}, x, **apply_kwargs)

    def predictive_mean_var(self, x: Array, **apply_kwargs) -> Tuple[Array, Array]:
        """(mean, variance) with zero predictive variance at the parameter level."""
        mean = self.predict(x, **apply_kwargs)
        var = jnp.zeros_like(mean)
        return mean, var


def _make_tx(cfg: MAPConfig) -> optax.GradientTransformation:
    # LR schedule (constant for now, but keep it a transform to swap later)
    schedule = cfg.learning_rate

    # Base optimizer
    if cfg.optimizer.lower() == "adam":
        base = optax.adam(schedule)
    elif cfg.optimizer.lower() == "sgd":
        base = optax.sgd(schedule, momentum=0.0, nesterov=False)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Optional decoupled weight decay
    wd = (
        optax.add_decayed_weights(cfg.weight_decay)
        if cfg.weight_decay > 0
        else optax.identity()
    )

    # Optional grad clipping
    clip = (
        optax.clip_by_global_norm(cfg.clip_grad_norm)
        if cfg.clip_grad_norm
        else optax.identity()
    )

    return optax.chain(clip, wd, base)


def _create_train_state(
    rng: Array,
    flax_module: nn.Module,
    init_input: Array,
    tx: optax.GradientTransformation,
) -> train_state.TrainState:
    variables = flax_module.init(rng, init_input)
    params = variables["params"]
    return train_state.TrainState.create(
        apply_fn=flax_module.apply, params=params, tx=tx
    )


def fit_map(
    *,
    logposterior_fn: Callable[[Dict[str, Any], Dict[str, Array]], Array],
    network: nn.Module,
    train_ds: Dict[str, Array],
    cfg: MAPConfig,
) -> PosteriorMAP:
    """Estimate MAP params by maximizing a user-provided log-posterior.

    Args:
        logposterior_fn: (params, batch) -> scalar log-posterior (sum or mean over batch).
                         You define model likelihood/prior inside this function.
        network: Flax module used only for initialization & apply.
        train_ds: dict with "x" and "y" arrays.
        cfg: training configuration.

    Returns:
        PosteriorMAP: wraps the point-estimate params + apply_fn.
    """

    # Negative log-posterior (we minimize)
    def nlp(params, batch):
        return -logposterior_fn(params, batch)

    # Single step (jit-able)
    nlp_and_grad = jax.value_and_grad(nlp)

    def train_step(state, batch):
        loss, grads = nlp_and_grad(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def run_epoch(state, batches):
        def step_fn(carry, batch):
            state = carry
            state, loss = train_step(state, batch)
            return state, loss

        state, losses = jax.lax.scan(step_fn, state, batches)
        # return last state and mean loss (over steps, then mean across batch dims if any)
        return state, jnp.mean(losses)

    # Init state
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    tx = _make_tx(cfg)
    state = _create_train_state(init_rng, network, train_ds["x"][: cfg.batch_size], tx)

    # Pre-batch to enable scan
    dataset = NumpyDataset(train_ds["x"], train_ds["y"])
    loader = NumpyLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    batches = list(loader)
    if len(batches) == 0:
        raise ValueError("Empty loader: check batch_size vs dataset size.")
    # From a list of dicts to dict of stacked arrays with leading 'num_batches'
    batches = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *batches)

    # Train
    for _ in range(cfg.num_epochs):
        state, _ = run_epoch(state, batches)

    return PosteriorMAP(state.params, apply_fn=network.apply)


# Backwards-compatible surface


def create_train_state(
    rng: Array,
    flax_module: nn.Module,
    init_input: Array,
    learning_rate: float,
    optimizer: Optional[str] = "adam",
) -> train_state.TrainState:
    """Kept for backward-compat; prefer _create_train_state + MAPConfig."""
    tx = _make_tx(MAPConfig(learning_rate=learning_rate, optimizer=optimizer or "adam"))
    return _create_train_state(rng, flax_module, init_input, tx)


def train_fn(
    logposterior_fn: Callable,
    network: nn.Module,
    train_ds: Dict[str, Array],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    rng_key: Array,  # kept for signature parity (unused; see cfg.seed)
    optimizer: str = "adam",
):
    """Backward-compatible wrapper: returns MAP params.

    Prefer `fit_map(logposterior_fn=..., network=..., train_ds=..., cfg=MAPConfig(...))`.
    """
    posterior = fit_map(
        logposterior_fn=logposterior_fn,
        network=network,
        train_ds=train_ds,
        cfg=MAPConfig(
            learning_rate=learning_rate,
            optimizer=optimizer,
            batch_size=batch_size,
            num_epochs=num_epochs,
            seed=int(
                jax.random.randint(rng_key, (), 0, 2**31 - 1)
            ),  # keep rng influence
        ),
    )
    return posterior.params
