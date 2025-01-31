import argparse
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from experiments import Experiment, load_experiment
from jax import Array
from jax.flatten_util import ravel_pytree
from tqdm import tqdm


def vmapped_ravel_pytree(pytree):
    return jax.vmap(lambda tree: ravel_pytree(tree)[0])(pytree)


def SteinIMQ(x: Array, sx: Array, y: Array, sy: Array, lengthscale: float):
    """ "Stein kernel with an IMQ base kernel."""
    d = len(x)
    sqdist = jnp.sum((x - y) ** 2)
    qf = 1.0 / (1.0 + sqdist / lengthscale**2)
    t3 = jnp.dot(sx, sy) * jnp.sqrt(qf)
    t2 = (1.0 / lengthscale**2) * (d + jnp.dot(sx - sy, x - y)) * qf ** (3 / 2)
    t1 = (-3.0 / lengthscale**4) * sqdist * qf ** (5 / 2)
    return t1 + t2 + t3


def GramSteinIMQ(y: Array, grad_y: Array, lengthscale: float):
    _, d = y.shape
    linv = 1.0 / lengthscale**2

    sqdist = jax.vmap(
        lambda x: jax.vmap(lambda y: jnp.clip(x @ x + y @ y - 2 * x @ y, a_min=0))(y)
    )(y)
    qf = 1.0 + sqdist * linv

    Spx = jnp.matmul(grad_y, y.T)
    Spxy = jnp.diag(Spx)[:, None] - Spx

    beta0 = 0.5
    beta1 = beta0 + 1.0
    beta2 = beta0 + 2.0

    t1 = -4.0 * beta0 * beta1 * linv * linv * sqdist / qf**beta2
    t2 = 2.0 * beta0 * linv * (d + Spxy + Spxy.T) / qf**beta1
    t3 = jnp.matmul(grad_y, grad_y.T) / qf**beta0
    return t1 + t2 + t3


def SlicedSteinIMQ(w, sw, v, sv, r_dot_g, lengthscale: float):
    """Sliced kernel with an IMQ base kernel."""
    sqdist = (w - v) ** 2
    qf = 1.0 / (1.0 + sqdist / lengthscale**2)

    t1 = sw * jnp.sqrt(qf) * sv
    t2 = (
        (1.0 / lengthscale**2)
        * r_dot_g
        * (r_dot_g + (sw - sv) * (w - v))
        * qf ** (3 / 2)
    )
    t3 = (-3.0 / lengthscale**4) * r_dot_g * r_dot_g * sqdist * qf ** (5 / 2)
    return t1 + t2 + t3


def KSD(x: Array, score_p: Array, lengthscale: float):
    """Kernelized Stein Discrepancy with an IMQ base kernel."""
    kp = GramSteinIMQ(x, score_p, lengthscale)
    ksd = jnp.sqrt(jnp.sum(kp)) / x.shape[0]
    return ksd


def SlicedKSD(x: Array, score_p: Array, r: Array, g: Array, lengthscale: float):
    """Sliced Kernelized Stein Discrepancy with an IMQ base kernel."""
    projected_samples = jnp.matmul(g, x.T)  # [n_slices, n_samples]
    projected_scores = jnp.matmul(r, score_p.T)  # [n_slices, n_samples]
    R_dot_G = jnp.sum(r * g, axis=1)

    kp_fn = jax.tree_util.Partial(SlicedSteinIMQ, lengthscale=lengthscale)
    Stein_matrices = jax.vmap(
        lambda a, b: jax.vmap(lambda c, d, e: kp_fn(a, b, c, d, e))(
            projected_samples, projected_scores, R_dot_G
        )
    )(projected_samples, projected_scores)
    sliced_ksds = jax.tree_util.tree_map(
        lambda mat: jnp.sum(mat) / x.shape[0] ** 2, Stein_matrices
    )
    return jnp.sqrt(jnp.mean(sliced_ksds))


def get_logprob_fn(loglikelihood_fn: Callable, logprior_fn: Callable):
    """Gives the log posterior."""

    def logprob_fn(parameters, data):
        X_train, y_train = data
        logprior = logprior_fn(parameters)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, (None, 0))(
            parameters, (X_train, y_train)
        )
        return logprior + jnp.sum(batch_loglikelihood)

    return logprob_fn


def randh(rng_key: Array, n_samples: int, dim: int):
    """Generates a smaple from the uniform distribution over the d-hypersphere."""
    x = jax.random.normal(rng_key, (n_samples, dim))
    y = x / jnp.sqrt(jnp.sum(x**2, axis=1))[:, None]
    return y


def get_number_of_params(experiment: Experiment):
    """Get the number of parameters in the neural network"""
    _, _, _, X_test, _, _ = experiment.load_data_fn(0)
    rng_key = jax.random.PRNGKey(0)
    init_positions = experiment.network().init(rng_key, X_test)["params"]
    vec_init_positions, _ = ravel_pytree(init_positions)
    return len(vec_init_positions)


def compute_discrepancies(
    experiment: Experiment,
    files: list,
    r: Array,
    g: Array,
    med_ksd: float,
    med_sksd: float,
):
    num_files = len(files)

    loglikelihood_fn = experiment.loglikelihood_fn
    logprior_fn = experiment.logprior_fn
    logprob_fn = get_logprob_fn(loglikelihood_fn, logprior_fn)
    load_data_fn = experiment.load_data_fn

    _, _, _, X_test, _, _ = experiment.load_data_fn(0)

    rng_key = jax.random.PRNGKey(0)
    init_positions = experiment.network().init(rng_key, X_test)["params"]
    _, unravel_fn = ravel_pytree(init_positions)

    # Compute discrepancies
    ksds = jnp.zeros(num_files)
    sksds = jnp.zeros(num_files)
    indices = [str(file).split("_")[-1].split(".")[0] for file in files]

    for j, file in tqdm(enumerate(files)):
        dataset_idx = int(str(file).split("_")[-1].split(".")[0])
        X_train, _, y_train, _, _, _= load_data_fn(dataset_idx=dataset_idx)
        X_train, y_train = jnp.array(X_train), jnp.array(y_train)

        score_fn = jax.jit(jax.grad(logprob_fn, argnums=0))

        data = jnp.load(file)
        samples = data["positions"]

        samples_dict = jax.vmap(unravel_fn, 0)(samples)
        grads_dict = jax.vmap(score_fn, (0, None))(samples_dict, (X_train, y_train))

        grads = vmapped_ravel_pytree(grads_dict)

        ksd = KSD(x=samples, score_p=grads, lengthscale=med_ksd)
        ksds = ksds.at[j].set(ksd)

        sksd = SlicedKSD(x=samples, score_p=grads, r=r, g=g, lengthscale=med_sksd)
        sksds = sksds.at[j].set(sksd)

    return ksds, sksds, indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        type=str,
        help="Path to where all the data of a given experiment are stored",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        help="Path to where all discrepancies will be saved",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the whole experiment"
    )
    parser.add_argument("--experiment", type=int, choices=np.arange(10))
    parser.add_argument("--algorithm", type=str, help="Name of the algorithm")
    args = parser.parse_args()

    experiment = load_experiment(args.experiment)

    WORKDIR = Path(args.workdir)
    SAVEDIR = Path(args.savedir)
    DATADIR = WORKDIR / f"seed_{args.seed}" / experiment.name / args.algorithm
    SAVEDIR = SAVEDIR / f"seed_{args.seed}" / experiment.name / args.algorithm

    median_heuristics = np.load(
        WORKDIR / f"seed_{args.seed}" / experiment.name / "median_heuristics.npz"
    )
    med_ksd, med_sksd = median_heuristics["med_ksd"], median_heuristics["med_sksd"]

    num_params = get_number_of_params(experiment)

    rng_key = jax.random.PRNGKey(12346)
    r = randh(rng_key, 1000, num_params)
    _, rng_key = jax.random.split(rng_key)
    g = randh(rng_key, 1000, num_params)

    for leaf in sorted(DATADIR.rglob("lr_*")):
        filenames = sorted(leaf.glob("saved_data*"))
        learning_rate = leaf.name
        if len(filenames) > 0:
            ksds, sksds, indices = compute_discrepancies(
                experiment, filenames, r, g, med_ksd, med_sksd
            )

            TARGETDIR = SAVEDIR / learning_rate
            TARGETDIR.mkdir(parents=True, exist_ok=True)

            jnp.savez(
                TARGETDIR / "discrepancies.npz",
                ksd=ksds,
                sksd=sksds,
                indices=indices,
            )
