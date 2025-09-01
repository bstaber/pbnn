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

import pbnn.mcmc as mcmc
from pbnn.utils.misc import thinning_fn

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", default=".", help="Directory where data will be stored")
flags.DEFINE_string("algorithm", default="sgld", help="name of the sgmcmc algorithm")
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
flags.DEFINE_float("step_size", 1e-8, "Step size of the algorithm")
flags.DEFINE_integer(
    "seed", default=0, help="Initial seed that will be split accross the functions"
)
flags.DEFINE_string(
    "init_method", default="map", help="Initialization method for MCMC."
)


def setup_directories(workdir, seed, experiment, algorithm, step_size):
    # root = os.path.join(
    #     workdir, f"seed_{seed}", experiment.name, algorithm.__name__, f"lr_{step_size}"
    # )
    # init_params_dir = os.path.join(
    #     workdir, f"seed_{seed}", experiment.name, "init_params"
    # )

    # if not os.path.isdir(root):
    #     os.makedirs(root)

    WORKDIR = Path(workdir)
    ROOT = (
        WORKDIR
        / f"seed_{seed}"
        / experiment.name
        / algorithm.__name__
        / f"lr_{step_size}"
    )
    INIT_PARAMS_DIR = WORKDIR / f"seed_{seed}" / experiment.name / "init_params"

    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT, INIT_PARAMS_DIR


def main(argv):
    """Main function for running SGMCMC algorithms and save results.

    Args:
        argv (list): Command line arguments.
    """
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    workdir = FLAGS.workdir
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed
    init_method = FLAGS.init_method

    batch_size = 32

    experiment = load_experiment(index=FLAGS.experiment)
    algorithm = getattr(mcmc, FLAGS.algorithm)

    ROOT, INIT_PARAMS_DIR = setup_directories(
        workdir, seed, experiment, algorithm, step_size
    )

    def sgmcmc_fn(algorithm, X, y, init_positions, map_positions, rng_key):
        """Helper function that runs a SGMCMC algorithm.

        Parameters
        ----------

        X
            Input features
        y
            Output features
        init_positions
            Initial positions for the SGMCMC algorithm
        rng_key
            PRNGKey

        Returns:
        -------
        Markov Chain of positions with burnin already removed

        """
        match algorithm.__name__:
            case "sgld":
                hparams = {"num_iterations": 100_00}
                burnin = 50_00
            case "pSGLD":
                # overwrite the init_positions in the case of pSGLD
                # initializating pSGLD close to a MAP estimation yields bad performances
                # init_positions = experiment.network().lazy_init(
                #     rng_key, jax.ShapeDtypeStruct((1, experiment.in_feats), jnp.float32)
                # )["params"]
                # _, rng_key = jr.split(rng_key)
                hparams = {"preconditioning_factor": 0.95, "num_iterations": 100_000}
                burnin = 50_000
            case "cyclical_sgld":
                hparams = {
                    "num_cycles": 5,
                    "num_sgd_steps": 10_000,
                    "num_sgld_steps": 20_000,
                    "burnin_sgld": 10_000,
                }
                burnin = 0
            case "sghmc":
                hparams = {"num_integration_steps": 10, "num_iterations": 10_000}
                burnin = 5_000
            case "sgld_cv":
                hparams = {
                    "num_iterations": 100_000,
                    "centering_positions": map_positions,
                }
                burnin = 50_000
            case "sghmc_cv":
                hparams = {
                    "num_iterations": 10_000,
                    "num_integration_steps": 10,
                    "centering_positions": map_positions,
                }
                burnin = 5_000
            case "sgld_svrg":
                hparams = {
                    "num_cv_iterations": 100,
                    "num_svrg_iterations": 1000,
                    "centering_positions": map_positions,
                }
                burnin = 50_000
            case "sghmc_svrg":
                hparams = {
                    "num_integration_steps": 10,
                    "num_cv_iterations": 100,
                    "num_svrg_iterations": 1_000,
                    "centering_positions": map_positions,
                }
                burnin = 50_000

        positions, ravel_fn, predict_fn = algorithm(
            X=X,
            y=y,
            loglikelihood_fn=experiment.loglikelihood_fn,
            logprior_fn=logprior_fn,
            init_positions=init_positions,
            batch_size=batch_size,
            step_size=step_size,
            rng_key=rng_key,
            **hparams,
        )

        # remove burnin
        positions = jax.tree_util.tree_map(lambda xx: xx[burnin::], positions)
        return positions, ravel_fn, predict_fn

    keys = jr.split(jr.PRNGKey(seed), num_datasets)
    for i, rng_key in track(enumerate(keys), total=num_datasets):
        # load i-th dataset
        X, _, y, X_test, _, _ = experiment.load_data_fn(i)

        # get initial positions and other hparams
        # load the MAP estimation for the i-th dataset
        # os.path.join(init_params_dir, f"init_params_{i}.pkl"),
        with open(INIT_PARAMS_DIR / f"init_params_{i}.pkl", "rb") as f:
            map_positions = pickle.load(f)

        if init_method == "map":
            init_positions = map_positions
        else:
            init_positions = experiment.network().lazy_init(
                rng_key, jax.ShapeDtypeStruct((1, experiment.in_feats), jnp.float32)
            )["params"]
            _, rng_key = jr.split(rng_key)

        # run the sgmcmc method
        # mcmc_key, noise_key = jr.split(rng_key)
        t_initial = time()
        positions, ravel_fn, predict_fn = sgmcmc_fn(
            algorithm,
            X,
            y,
            init_positions,
            map_positions,
            rng_key,
        )
        t_final = time()

        # check if the algorithm as failed
        isnan = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.isnan(x)), positions)
        success = sum(jax.tree_util.tree_leaves(isnan)) == 0

        if success:
            # thin the MCMC output
            rpositions = ravel_fn(positions)
            idx = thinning_fn(rpositions, size=2000)
            positions = jax.tree_util.tree_map(lambda xx: xx[idx], positions)

            # predict
            f_predictions = predict_fn(
                experiment.network(), positions, X_test
            ).squeeze()

            # generate the noisy predictions
            # y_predictions = f_predictions + experiment.noise_level * jr.normal(
            #     noise_key, shape=(len(f_predictions), 1)
            # )

            jnp.savez(
                ROOT / f"saved_data_{i}.npz",
                positions=rpositions[idx],
                predictions=f_predictions,
                time=t_final - t_initial,
            )
        else:
            print("Failed!")


if __name__ == "__main__":
    app.run(main)
