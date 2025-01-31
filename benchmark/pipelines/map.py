import pickle
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from absl import app, flags, logging
from experiments import load_experiment
from logprobs import logprior_fn
from rich.progress import track

from pbnn.map_estimation import train_fn

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", ".", "Directory where data will be stored")
flags.DEFINE_integer(
    "experiment", 1, "Index of the experiment", lower_bound=1, upper_bound=10
)
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_epochs", 10_000, "Total number of iterations to perform")
flags.DEFINE_integer(
    "num_datasets",
    1_00,
    "Number of datasets used to estimate the coverage probability",
)
flags.DEFINE_float("step_size", 1e-2, "Step size of the algorithm")
flags.DEFINE_integer(
    "seed", default=0, help="Initial seed that will be split accross the functions"
)


def logposterior_estimator_fn(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Log posterior function"""

    def logposterior_fn(parameters, data_batch):
        logprior = logprior_fn(parameters)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, in_axes=(None, 0))
        return logprior + data_size * jnp.mean(
            batch_loglikelihood(parameters, data_batch), axis=0
        )

    return logposterior_fn


def setup_directories(workdir, seed, experiment, alg):
    # root = os.path.join(FLAGS.workdir, f"seed_{seed}", experiment.name, alg)
    #     if not os.path.isdir(root):
    #         os.makedirs(root)
    ROOT = Path(workdir) / f"seed_{seed}" / experiment.name / alg
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def main(argv):
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed

    experiment = load_experiment(FLAGS.experiment)
    load_data_fn = experiment.load_data_fn
    loglikelihood_fn = experiment.loglikelihood_fn
    network = experiment.network
    data_size = experiment.data_size

    logposterior_fn = logposterior_estimator_fn(
        logprior_fn, loglikelihood_fn, data_size
    )

    keys = jr.split(jr.PRNGKey(seed), num_datasets)

    # Computing MAPs for initializing the MCMC methods
    alg = "init_params"
    ROOT = setup_directories(FLAGS.workdir, seed, experiment, alg)

    for i, key in track(enumerate(keys), total=len(keys)):
        X_train, _, y_train, _, _, _ = load_data_fn(dataset_idx=i)
        X_train, y_train = jnp.array(X_train), jnp.array(y_train)

        network = experiment.network()

        params = train_fn(
            logposterior_fn=logposterior_fn,
            network=network,
            train_ds={"x": X_train, "y": y_train},
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=step_size,
            rng_key=key,
        )

        # with open(os.path.join(root, f"init_params_{i}.pkl"), "wb") as f:
        with open(ROOT / f"init_params_{i}.pkl", "wb") as f:
            pickle.dump(params, f)


if __name__ == "__main__":
    app.run(main)
