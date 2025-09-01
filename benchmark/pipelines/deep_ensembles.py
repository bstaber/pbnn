from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
from absl import app, flags, logging
from experiments import load_experiment
from logprobs import logprior_fn
from rich.progress import track

from pbnn.deep_ensembles import deep_ensembles_fn

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", default=".", help="Directory where data will be stored")
flags.DEFINE_integer(
    "experiment",
    default=1,
    help="Index of the experiment",
    lower_bound=1,
    upper_bound=10,
)
flags.DEFINE_integer(
    "num_datasets",
    default=100,
    help="Total number of training datasets",
)
flags.DEFINE_float("step_size", default=1e-2, help="Step size of the algorithm")
flags.DEFINE_integer(
    "seed", default=0, help="Initial seed that will be split accross the functions"
)


def setup_directories(workdir, seed, experiment, step_size):
    # root = os.path.join(
    #     workdir, f"seed_{seed}", experiment.name, "deep_ensembles", f"lr_{step_size}"
    # )

    # if not os.path.isdir(root):
    #     os.makedirs(root)
    WORKDIR = Path(workdir)
    ROOT = (
        WORKDIR
        / f"seed_{seed}"
        / experiment.name
        / "deep_ensembles"
        / f"lr_{step_size}"
    )

    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def main(argv):
    """Main function to train deep ensembles and save results.

    Args:
        argv (list): Command line arguments.
    """
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    workdir = FLAGS.workdir
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed

    ensemble_size = 200
    batch_size = 32

    experiment = load_experiment(index=FLAGS.experiment)

    ROOT = setup_directories(workdir, seed, experiment, step_size)

    keys = jr.split(jr.PRNGKey(FLAGS.seed), num_datasets)
    for i, rng_key in track(enumerate(keys), total=num_datasets):
        # load i-th dataset
        # i, rng_key = dataset_idx, keys[dataset_idx]
        X, _, y, X_test, _, _ = experiment.load_data_fn(i)

        # get network
        network = experiment.network()

        # apply deep_ensembles
        # ens_key, noise_key = jr.split(rng_key)

        t_initial = time()
        positions, ravel_fn, predict_fn = deep_ensembles_fn(
            X,
            y,
            experiment.loglikelihood_fn,
            logprior_fn,
            network,
            batch_size,
            10_000,
            step_size,
            ensemble_size,
            rng_key,
        )
        t_final = time()

        # predict
        f_predictions = predict_fn(network, positions, X_test).squeeze()

        # noisy observations
        # y_predictions = f_predictions + experiment.noise_level * jr.normal(
        #     noise_key, shape=(len(f_predictions), 1)
        # )

        jnp.savez(
            ROOT / f"saved_data_{i}.npz",
            positions=ravel_fn(positions),
            predictions=f_predictions,
            time=t_final - t_initial,
        )


if __name__ == "__main__":
    app.run(main)
