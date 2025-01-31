import pickle
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
from absl import app, flags, logging
from experiments import load_experiment
from logprobs import logprior_fn
from rich.progress import track

from pbnn.swag import swag_fn

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
    default=1_00,
    help="Number of datasets used to estimate the coverage probability",
)
flags.DEFINE_float("step_size", default=1e-2, help="Step size of the algorithm")
flags.DEFINE_integer(
    "seed", default=0, help="Initial seed that will be split accross the functions"
)


def setup_directories(workdir, seed, experiment, step_size):
    # root = os.path.join(
    #     workdir, f"seed_{seed}", experiment.name, "swag", f"lr_{step_size}"
    # )
    # if not os.path.isdir(root):
    #     os.makedirs(root)
    # init_params_dir = os.path.join(
    #     workdir, f"seed_{seed}", experiment.name, "init_params"
    # )
    ROOT = Path(workdir) / f"seed_{seed}" / experiment.name / "swag" / f"lr_{step_size}"
    INIT_PARAMS_DIR = Path(workdir) / f"seed_{seed}" / experiment.name / "init_params"

    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT, INIT_PARAMS_DIR


def main(argv):
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    workdir = FLAGS.workdir
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed

    num_epochs = 2_000
    batch_size = 32
    cov_rank = 100

    experiment = load_experiment(index=FLAGS.experiment)

    ROOT, INIT_PARAMS_DIR = setup_directories(workdir, seed, experiment, step_size)

    keys = jr.split(jr.PRNGKey(FLAGS.seed), num_datasets)
    for i, rng_key in track(enumerate(keys), total=num_datasets):
        # load i-th dataset
        X, _, y, X_test, _, _ = experiment.load_data_fn(i)

        # with open(os.path.join(init_params_dir, f"init_params_{i}.pkl"), "rb") as f:
        with open(INIT_PARAMS_DIR / f"init_params_{i}.pkl", "rb") as f:
            map_params = pickle.load(f)

        # apply swag
        # train_key, noise_key = jr.split(rng_key)

        t_initial = time()
        positions, ravel_fn, predict_fn = swag_fn(
            X,
            y,
            experiment.loglikelihood_fn,
            logprior_fn,
            experiment.network(),
            map_params,
            batch_size,
            num_epochs,
            step_size,
            cov_rank,
            rng_key,
        )
        t_final = time()

        # predict
        f_predictions = predict_fn(experiment.network(), positions, X_test)

        # noisy predictions
        # y_predictions = f_predictions + experiment.noise_level * jr.normal(
        #     noise_key, shape=(len(f_predictions), 1)
        # )

        # os.path.join(root, f"saved_data_{i}.npz"),
        jnp.savez(
            ROOT / f"saved_data_{i}.npz",
            positions=ravel_fn(positions),
            predictions=f_predictions,
            time=t_final - t_initial,
        )


if __name__ == "__main__":
    app.run(main)
