from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
from absl import app, flags, logging
from experiments import load_experiment
from jax.flatten_util import ravel_pytree
from logprobs import logprior_fn
from rich.progress import track

from pbnn.mcmc import hmc

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
flags.DEFINE_float("step_size", 1e-6, "Step size of the algorithm")
flags.DEFINE_integer(
    "seed", default=0, help="Initial seed that will be split accross the functions"
)


def setup_directories(workdir, seed, experiment):
    # root = os.path.join(workdir, f"seed_{seed}", experiment.name, "hmc")
    # if not os.path.isdir(root):
    #     os.makedirs(root)
    directory_path = Path(workdir) / f"seed_{seed}" / experiment.name / "hmc"
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def main(argv):
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    workdir = FLAGS.workdir
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed

    experiment = load_experiment(index=FLAGS.experiment)
    load_data_fn = experiment.load_data_fn
    network = experiment.network()
    loglikelihood_fn = experiment.loglikelihood_fn

    # Define full-batch logprob_fn
    def logprob_fn(parameters):
        logprior = logprior_fn(parameters)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, (None, 0))(parameters, (X, y))
        return logprior + jnp.sum(batch_loglikelihood)

    keys = jr.split(jr.PRNGKey(FLAGS.seed), num_datasets)
    for dataset_idx, rng_key in track(enumerate(keys), total=num_datasets):
        X, _, y, X_test, _, _ = load_data_fn(dataset_idx=dataset_idx)

        # rng_key = jr.PRNGKey(0)
        rng_keys = jr.split(rng_key)
        init_positions = network.init(rng_keys[0], X)["params"]
        num_params = len(ravel_pytree(init_positions)[0])

        # Run 3 HMC chains and concatenate the results
        # rng_key = jr.PRNGKey(data<et_idx * (FLAGS.seed + 1))

        concat_positions = []
        concat_predictions = []

        chains_keys = jr.split(rng_keys[1], 3)
        for key in chains_keys:
            init_positions = network.init(key, X)["params"]
            _, rng_key = jr.split(key)

            t_initial = time()
            positions, ravel_fn, predict_fn = hmc(
                logprob_fn=logprob_fn,
                init_positions=init_positions,
                num_samples=200,
                step_size=step_size,
                inverse_mass_matrix=jnp.ones(num_params),
                num_integration_steps=10_000,
                rng_key=key,
            )
            t_final = time()

            predictions = predict_fn(network, positions, X_test).squeeze()

            concat_positions += [ravel_fn(positions)]
            concat_predictions += [predictions]

        concat_positions = jnp.stack(concat_positions, axis=0)
        concat_predictions = jnp.stack(concat_predictions, axis=0)

        # Save data
        ROOT = setup_directories(workdir, seed, experiment)

        jnp.savez(
            ROOT / f"saved_data_{dataset_idx}.npz",
            positions=concat_positions,
            predictions=concat_predictions,
            time=t_final - t_initial,
        )


if __name__ == "__main__":
    app.run(main)
