import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from experiments import load_experiment
from jax.config import config
from tqdm import tqdm

config.update("jax_enable_x64", True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from run_bayesian_algorithms import dropouts, step_sizes


@jax.jit
def GramDistKernel(X: jnp.ndarray, Y: jnp.ndarray):
    return jax.vmap(
        lambda x: jax.vmap(
            lambda y: jnp.sqrt(x @ x)
            + jnp.sqrt(y @ y)
            - jnp.sqrt(jnp.clip(x @ x + y @ y - 2 * x @ y, a_min=0.0))
        )(Y)
    )(X)


def MaximumMeanDiscrepancy(X: jnp.ndarray, Y: jnp.ndarray):
    kxx = GramDistKernel(X, X)
    kyy = GramDistKernel(Y, Y)
    kxy = GramDistKernel(X, Y)
    mmd = (
        (1.0 / X.shape[0] ** 2) * jnp.sum(kxx)
        + (1.0 / Y.shape[0] ** 2) * jnp.sum(kyy)
        - (2.0 / (X.shape[0] * Y.shape[0])) * jnp.sum(kxy)
    )
    return mmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=int,
        choices=np.arange(10),
    )
    parser.add_argument(
        "--workdir", type=str, help="Path to where all the heavy data are saved"
    )
    parser.add_argument(
        "--savedir", type=str, help="Path to where all the light data are saved"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the whole experiment"
    )
    parser.add_argument(
        "--dataset_idx", type=int, default=0, help="Index of the training dataset"
    )
    args = parser.parse_args()

    # load experiment
    experiment = load_experiment(index=args.experiment)

    # root = os.path.join(args.workdir, f"seed_{args.seed}", experiment.name)
    ROOT = Path(args.workdir) / f"seed_{args.seed}" / experiment.name
    ROOT.mkdir(parents=True, exist_ok=True)

    samples = dict()
    samples_list = list()

    algorithms = [
        "cyclical_sgld",
        "swag",
        "pSGLD",
        "sgld",
        "sgld_cv",
        "sgld_svrg",
        "sghmc",
        "sghmc_cv",
        "sghmc_svrg",
        "deep_ensembles",
        "laplace",
    ]
    for i, algorithm in tqdm(enumerate(algorithms)):
        for j, learning_rate in tqdm(enumerate(step_sizes[algorithm])):
            # datadir = os.path.join(root, algorithm, f"lr_{learning_rate}")
            DATADIR = ROOT / algorithm / f"lr_{learning_rate}"
            # file = os.path.join(datadir, f"saved_data_{args.dataset_idx}.npz")
            file = DATADIR / f"saved_data_{args.dataset_idx}.npz"
            if os.path.isfile(file):
                samples[algorithms[i] + f"_{j}"] = file
                samples_list.append({"algorithm": algorithms[i], "step_size": j})

    algorithm = "mcdropout"
    for j, learning_rate in tqdm(enumerate(step_sizes[algorithm])):
        for irate, rate in tqdm(enumerate(dropouts)):
            # datadir = os.path.join(root, algorithm, f"lr_{learning_rate}", f"rate_{rate}")
            DATADIR = ROOT / algorithm / f"lr_{learning_rate}" / f"rate_{rate}"
            # file = os.path.join(datadir, f"saved_data_{args.dataset_idx}.npz")
            file = DATADIR / f"saved_data_{args.dataset_idx}.npz"
            if os.path.isfile(file):
                samples[algorithm + f"{irate}" + f"_{j}"] = file
                samples_list.append(
                    {"algorithm": algorithm, "step_size": j, "rate": irate}
                )

    nsim = len(samples.keys())
    print(f"len(samples.keys()): {nsim}")
    print(f"len(samples_list): {len(samples_list)}")

    # path_to_hmc = os.path.join(root, "hmc", f"saved_data_{args.dataset_idx}.npz")
    path_to_hmc = ROOT / "hmc" / f"saved_data_{args.dataset_idx}.npz"
    data_hmc = jnp.load(path_to_hmc)

    MMD_params = jnp.zeros((nsim, nsim))
    MMD_preds = jnp.zeros((nsim, nsim))
    MMD_params_hmc = jnp.zeros(nsim)
    MMD_preds_hmc = jnp.zeros(nsim)

    preds_hmc = []
    params_hmc = []
    for chain in range(3):
        preds_hmc += [data_hmc["predictions"][chain][100:]]
        params_hmc += [data_hmc["positions"][chain][100:]]
    preds_hmc = np.concatenate(preds_hmc, axis=0).squeeze()
    params_hmc = np.concatenate(params_hmc, axis=0).squeeze()
    
    if preds_hmc.ndim == 3:
        preds_hmc = preds_hmc[:, :, 0]
        
    for i, (ikey, ival) in tqdm(enumerate(samples.items())):
        data_i = jnp.load(ival)
        x_params = jnp.array(data_i["positions"], dtype=jnp.float64)
        x_preds = jnp.array(data_i["predictions"].squeeze(), dtype=jnp.float64)
        
        if x_preds.ndim == 3:
            x_preds = x_preds[:, :, 0]

        if x_params.ndim == 1:
            x_params = x_params[None, :]

        mmd_params_hmc = MaximumMeanDiscrepancy(X=x_params, Y=params_hmc)
        MMD_params_hmc = MMD_params_hmc.at[i].set(mmd_params_hmc)

        mmd_preds_hmc = MaximumMeanDiscrepancy(X=x_preds, Y=preds_hmc)
        MMD_preds_hmc = MMD_preds_hmc.at[i].set(mmd_preds_hmc)

        for j, (jkey, jval) in enumerate(samples.items()):
            data_j = jnp.load(jval)
            y_params = jnp.array(data_j["positions"], dtype=jnp.float64)
            y_preds = jnp.array(data_j["predictions"].squeeze(), dtype=jnp.float64)

            if y_preds.ndim == 3:
                y_preds = y_preds[:, :, 0]

            if y_params.ndim == 1:
                y_params = y_params[None, :]

            mmd_params = MaximumMeanDiscrepancy(X=x_params, Y=y_params)
            MMD_params = MMD_params.at[i, j].set(mmd_params)

            mmd_preds = MaximumMeanDiscrepancy(X=x_preds, Y=y_preds)
            MMD_preds = MMD_preds.at[i, j].set(mmd_preds)

    # savedir = os.path.join(
    #     args.savedir, f"seed_{args.seed}", experiment.name, "MMD_matrices_float64"
    # )
    # if not os.path.isdir(savedir):
    #     os.makedirs(savedir)
    SAVEDIR = (
        Path(args.savedir)
        / f"seed_{args.seed}"
        / experiment.name
        / "MMD_matrices_float64"
    )
    SAVEDIR.mkdir(parents=True, exist_ok=True)

    # os.path.join(savedir, f"MMD_Matrices_{args.dataset_idx}.npz")
    jnp.savez(
        SAVEDIR / f"MMD_Matrices_{args.dataset_idx}.npz",
        MMD_params=MMD_params,
        MMD_preds=MMD_preds,
        MMD_params_hmc=MMD_params_hmc,
        MMD_preds_hmc=MMD_preds_hmc,
    )

    # os.path.join(savedir, f"samples_list_{args.dataset_idx}.npz")
    jnp.savez(
        SAVEDIR / f"samples_list_{args.dataset_idx}.npz",
        samples_list=samples_list,
    )
