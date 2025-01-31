from pathlib import Path
from time import time
from typing import Union

import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from absl import app, flags
from experiments import load_experiment
from jax import Array
from rich.progress import track
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split

from pbnn.map_estimation import train_fn


class QuantileRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        model: nn.Module,
        alpha: float,
        batch_size: int = 32,
        num_epochs: int = 1000,
        lr: float = 5e-3,
        rng_key: Array = jr.PRNGKey(0),
    ):
        self.model = clone(model)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.alpha = alpha
        self.alpha_ = [alpha, 0.5, 1 - alpha]
        self.rng_key = rng_key

    def fit(self, X, y):
        X, y = jnp.array(X), jnp.array(y)

        def loss_fn(params, batch_of_data):
            X, target = batch_of_data
            preds = self.model.apply({"params": params}, X)

            losses = []
            preds_reshape = preds.reshape((-1, target.shape[1], len(self.alpha_)))
            for i, q in enumerate(self.alpha_):
                errors = target - preds_reshape[..., i]
                losses.append(jnp.maximum((q - 1) * errors, q * errors))

            loss = jnp.mean(jnp.sum(jnp.concatenate(losses, axis=-1), axis=-1))
            return -loss

        self.params = train_fn(
            logposterior_fn=loss_fn,
            network=self.model,
            train_ds={"x": X, "y": y},
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.lr,
            rng_key=self.rng_key,
        )

        _, self.rng_key = jr.split(self.rng_key)

        return self

    def predict(self, X):
        X = jnp.array(X)
        return self.model.apply({"params": self.params}, X).squeeze()


class SplitCQR(object):
    def __init__(
        self,
        regressor,
        X_train: Union[list, np.ndarray],
        y_train: np.ndarray,
        X_cal: Union[list, np.ndarray],
        y_cal: np.ndarray,
        alpha: float,
    ) -> object:
        self.alpha = alpha

        self.X_train = X_train
        self.y_train = y_train

        self.X_cal = X_cal
        self.y_cal = y_cal

        self.output_dim = y_train.shape[1]
        self.regressor = clone(regressor)
        self.__class__.__name__ = "split_cqr"

    def fit(self):
        self.regressor.fit(self.X_train, self.y_train)
        pred_cal = self.regressor.predict(self.X_cal).reshape(-1, self.output_dim, 3)

        r_lo = pred_cal[:, :, 0] - self.y_cal
        r_hi = self.y_cal - pred_cal[:, :, 2]
        self.residuals = np.max(np.stack([r_lo, r_hi], axis=2), axis=2)
        return self

    def predict(self, X_test: np.ndarray):
        n = len(self.X_cal)

        pred_test = self.regressor.predict(X_test)
        q_med, q_lo, q_hi = pred_test[:, 1::3], pred_test[:, 0::3], pred_test[:, 2::3]

        qleft = np.empty((len(X_test), self.output_dim))
        qright = np.empty((len(X_test), self.output_dim))
        for k in range(self.output_dim):
            q = np.quantile(
                self.residuals[:, k],
                (1 - self.alpha) / (1.0 + 1.0 / n),
                axis=0,
                method="higher",
            )
            qleft[:, k] = q_lo[:, k] - q
            qright[:, k] = q_hi[:, k] + q
        return q_med, qleft, qright


def setup_directories(workdir, seed, experiment, step_size):
    # root = os.path.join(
    #     workdir, f"seed_{seed}", experiment.name, "split_cqr", f"lr_{step_size}"
    # )

    # if not os.path.isdir(root):
    #     os.makedirs(root)
    ROOT = (
        Path(workdir)
        / f"seed_{seed}"
        / experiment.name
        / "split_cqr"
        / f"lr_{step_size}"
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def main(argv):
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "workdir", default=".", help="Directory where data will be stored"
    )
    flags.DEFINE_integer(
        "experiment",
        default=1,
        help="Index of the experiment",
        lower_bound=1,
        upper_bound=10,
    )
    flags.DEFINE_integer(
        "num_datasets",
        default=1_00,
        help="Number of datasets used to estimate the coverage probability",
    )
    flags.DEFINE_float("step_size", 1e-3, "Step size of the algorithm")
    flags.DEFINE_integer(
        "seed", default=0, help="Initial seed that will be split accross the functions"
    )

    workdir = FLAGS.workdir
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed
    batch_size = 32
    num_epochs = 10_000
    alphas = np.linspace(0.05, 0.95, 19)
    calibration_size = 0.2

    experiment = load_experiment(index=FLAGS.experiment)

    ROOT = setup_directories(workdir, seed, experiment, step_size)

    keys = jr.split(jr.PRNGKey(FLAGS.seed), num_datasets)
    for i, rng_key in track(enumerate(keys), total=num_datasets):
        # load i-th dataset
        X, _, y, X_test, _, _ = experiment.load_data_fn(i)

        # train_key, noise_key = jr.split(rng_key)

        f_pred, qlow, qhigh = [], [], []
        for alpha in alphas:
            regressor = QuantileRegressor(
                model=experiment.network(),
                alpha=alpha,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=step_size,
                rng_key=rng_key,
            )

            X_train, X_cal, y_train, y_cal = train_test_split(
                X,
                y,
                test_size=calibration_size,
            )
            model = SplitCQR(regressor, X_train, y_train, X_cal, y_cal, alpha)

            t_initial = time()
            model.fit()
            t_final = time()

            f_pred_, qlow_, qhigh_ = model.predict(X_test)
            f_pred.append(f_pred_)
            qlow.append(qlow_)
            qhigh.append(qhigh_)

        f_predictions = jnp.stack(f_pred, axis=1)
        qlow = jnp.stack(qlow, stack=1)
        qhigh = jnp.stack(qhigh, stack=1)

        if f_predictions.ndim == 1:
            f_predictions = f_predictions[:, None]

        # y_predictions = f_predictions + experiment.noise_level * jr.normal(
        #     noise_key, shape=(len(f_predictions), 1)
        # )

        # os.path.join(root, f"saved_data_{i}.npz"),
        jnp.savez(
            ROOT / f"saved_data_{i}.npz",
            predictions=f_predictions,
            qlow=qlow,
            qhigh=qhigh,
            time=t_final - t_initial,
        )


if __name__ == "__main__":
    app.run(main)
