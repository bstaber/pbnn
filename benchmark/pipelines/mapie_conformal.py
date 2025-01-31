from pathlib import Path
from time import time

import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import optax
from absl import app, flags
from experiments import load_experiment
from jax import Array
from mapie.regression import MapieRegressor
from rich.progress import track
from sklearn.base import BaseEstimator, RegressorMixin, clone

from pbnn.map_estimation import train_fn


class Regressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        num_epochs: int = 1000,
        lr: float = 5e-3,
        rng_key: Array = jr.PRNGKey(0),
    ):
        self.model = clone(model)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.rng_key = rng_key

    def fit(self, X, y):
        X, y = jnp.array(X), jnp.array(y)

        def loss_fn(params, batch_of_data):
            X_batch, y_batch = batch_of_data
            predictions = self.model.apply({"params": params}, X_batch).squeeze()
            return -optax.losses.l2_loss(predictions, y_batch).mean()

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


def setup_directories(workdir, seed, experiment, algorithm, step_size):
    # root = os.path.join(
    #         workdir, f"seed_{seed}", experiment.name, flags.algorithm, f"lr_{step_size}"
    #     )

    # if not os.path.isdir(root):
    #     os.makedirs(root)
    ROOT = (
        Path(workdir) / f"seed_{seed}" / experiment.name / algorithm / f"lr_{step_size}"
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def main(argv):
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "workdir", default=".", help="Directory where data will be stored"
    )
    flags.DEFINE_string(
        "algorithm",
        default="split_cp",
        help="name of the CP variant (cv_plus, split_cp)",
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
    num_folds = 10

    experiment = load_experiment(index=FLAGS.experiment)

    ROOT = setup_directories(workdir, seed, experiment, flags.algorithm, step_size)

    keys = jr.split(jr.PRNGKey(FLAGS.seed), num_datasets)
    for i, rng_key in track(enumerate(keys), total=num_datasets):
        # load i-th dataset
        X, y, X_test, y_test = experiment.load_data_fn(i)

        # train_key, noise_key = jr.split(rng_key)

        network = experiment.network()

        if flags.algorithm == "cv_plus":
            regressor = Regressor(
                model=network,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=step_size,
                rng_key=rng_key,
            )
            mapie = MapieRegressor(regressor, method="plus", cv=num_folds)
        elif flags.algorithm == "split_cp":
            regressor = Regressor(
                model=network,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=step_size,
                rng_key=rng_key,
            )
            mapie = MapieRegressor(regressor, method="base", cv="split")
        else:
            raise ValueError("wrong algorithm")

        t_initial = time()
        mapie.fit(X, y)
        t_final = time()

        f_predictions, y_pis = mapie.predict(X_test)
        qlow, qhigh = y_pis[:, 0, :], y_pis[:, 1, :]

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
