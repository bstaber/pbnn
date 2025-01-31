import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from experiments import load_experiment
from jax import Array
from scipy.spatial.distance import pdist


def median_heuristic(x: Array):
    """Computes the median heuristic for the given sample."""
    return np.median(pdist(x))


def randh(rng_key: Array, n_samples: int, dim: int):
    """Generates a smaple from the uniform distribution over the d-hypersphere."""
    x = jr.normal(rng_key, (n_samples, dim))
    y = x / jnp.sqrt(jnp.sum(x**2, axis=1))[:, None]
    return y


def get_data_files(folder: str):
    """Get all the .npz files the folder."""
    saved_data = sorted([path for path in Path(folder).rglob("*.npz")])
    return saved_data


def estimate_lengthscale(experiment: str, filenames: list, n: int = 1000):
    """Estimates the median heuristic with a subsample of the whole available data.
    It returns the median heuristics that should be used for computing KSD and SKSD.
    """

    bins = [0]
    repeated_filenames = []
    idx_map = []
    for filename in filenames:
        algorithm = str(filename).split(experiment)[1].split("/")[1]
        if algorithm == "hmc":
            pass
        elif algorithm == "deep_ensembles":
            bins += [200]
            repeated_filenames += [filename] * 200
            idx_map += [np.arange(200)]
        else:
            bins += [2000]
            repeated_filenames += [filename] * 2000
            idx_map += [np.arange(2000)]
    num_samples = sum(bins)
    idx = np.random.choice(num_samples, n)
    idx_map = np.concatenate(idx_map)

    # idx = np.random.choice(m*len(filenames), n)
    # bins = list(range(0,m*len(filenames)+m,m))
    # cats = np.digitize(idx, bins)

    print("Collecting samples")
    samples = []
    for i in idx:
        f = repeated_filenames[i]
        j = idx_map[i]
        data = jnp.load(f)
        if "positions" in data.files:
            x = data["positions"][j]
        else:
            raise KeyError(f"key positions not in this file: {f}")
        samples.append(x)
        if jnp.sum(jnp.isnan(x)) > 0 or jnp.sum(jnp.isinf(x)) > 0:
            print(
                f"type(x) = {type(x)}, x.shape = {x.shape}, isnan: {jnp.sum(jnp.isnan(x))}, file: {f}"
            )
    samples = jnp.stack(samples, axis=0)

    print(f"isnan: {jnp.sum(jnp.isnan(samples))}")

    rng_key = jr.PRNGKey(12345)
    g = randh(rng_key, n, samples.shape[1])

    projected_samples = jnp.sum(samples * g, axis=1)[:, None]

    med_ksd = median_heuristic(samples)
    med_sksd = median_heuristic(projected_samples)
    return med_ksd, med_sksd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=int,
        choices=np.arange(10),
    )
    parser.add_argument(
        "--workdir", type=str, help="Path to where all the data are saved"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the whole experiment"
    )

    args = parser.parse_args()

    # load experiment
    experiment = load_experiment(index=args.experiment)

    # gathering all the filenames
    # filenames = get_data_files(
    #     os.path.join(args.workdir, f"seed_{args.seed}", experiment.name)
    # )

    experiment_directory = Path(args.workdir) / f"seed_{args.seed}" / experiment.name
    filenames = get_data_files(experiment_directory)

    print(f"Number of files found: {len(filenames)}")
    print("Examples:")
    print(filenames[:5])

    # removing mcdropout from the list of files
    algorithms = [
        str(filename).split(experiment.name)[1].split("/")[1] for filename in filenames
    ]
    mcdpidx = np.where(np.array(algorithms) == "mcdropout")[0]
    filenames = np.delete(filenames, mcdpidx)

    # computing median heuristics
    if len(filenames) > 0:
        med_ksd, med_sksd = estimate_lengthscale(experiment.name, filenames, n=1000)

        # os.path.join(args.workdir, f"seed_{args.seed}", experiment.name, "median_heuristics.npz")
        np.savez(
            experiment_directory / "median_heuristics.npz",
            med_ksd=med_ksd,
            med_sksd=med_sksd,
        )
