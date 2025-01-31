import argparse
import json
import os
import shutil
import sys

import numpy as np
from rich.console import Console
from utils import barrier, launch

sys.path.append("pipelines")


step_sizes = {
    "laplace": np.logspace(-1, -4, 10)[::-1].tolist(),
    "pSGLD": np.logspace(-8, -6, 10)[::-1].tolist(),
    "sgld": np.logspace(-8, -5, 10).tolist(),
    "sgld_cv": np.logspace(-8, -5, 10).tolist(),
    "sgld_svrg": np.logspace(-8, -5, 10).tolist(),
    "sghmc": np.logspace(-8, -6, 10).tolist(),
    "sghmc_cv": np.logspace(-8, -6, 10).tolist(),
    "sghmc_svrg": np.logspace(-8, -6, 10).tolist(),
    "deep_ensembles": np.logspace(-1, -4, 10)[::-1].tolist(),
    "cyclical_sgld": np.logspace(-8, -5, 10).tolist(),
    "swag": np.logspace(-7, -5, 10).tolist(),
    "mcdropout": np.logspace(-1, -5, 10)[::-1].tolist(),
}

dropouts = np.linspace(0.1, 0.9, 9)


def remove_files_with_extension(directory, extension):
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)


def clean(flag=True):
    if flag:
        print("Removing log files *.o, *.e, and *.slurm")
        remove_files_with_extension(".", ".o")
        remove_files_with_extension(".", ".e")
        remove_files_with_extension(".", ".slurm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=0, choices=np.arange(10))
    parser.add_argument("--experiment", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="path to where the heavy data will be stored (weights)",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        required=True,
        help="path to where the light data will be stored (metrics)",
    )
    parser.add_argument("--template_folder", type=str, default="pipelines/templates")
    parser.add_argument("--command", type=str, default="sbatch")
    parser.add_argument("--num_datasets", type=int, default=100)
    parser.add_argument("--clean_slurm_outputs", action="store_true")

    args = parser.parse_args()

    console = Console(style="green")

    with open("slurm_params.json", "r") as f:
        slurm_params = json.load(f)
    common_params = slurm_params.copy()
    common_params["experiment"] = args.experiment
    # common_params["seed"] = args.seed
    common_params["workdir"] = args.workdir
    common_params["savedir"] = args.savedir
    common_params["template_folder"] = args.template_folder
    common_params["command"] = args.command
    common_params["time"] = int(60 * 60 * 12)
    common_params["num_datasets"] = args.num_datasets

    console.print("Running the whole benchmark")
    console.print("Common parameters are:")
    console.print(common_params)

    NUM_SEEDS = 10

    for seed in range(NUM_SEEDS):
        # Maximum a posteriori estimations
        barrier(100)
        params = common_params.copy()
        params["seed"] = seed
        res = launch(template_name="map", **params)
        console.print(f"Running map for seed {seed}")

    barrier(1)
    clean(args.clean_slurm_outputs)
    for seed in range(NUM_SEEDS):
        common_params["seed"] = seed

        # running laplace
        for step_size in step_sizes["laplace"]:
            params = common_params.copy()
            params["step_size"] = step_size
            params["algorithm"] = "laplace"
            barrier(100)
            res = launch(template_name="bayesian", **params)
            console.print(f"Running laplace with step size {step_size}")

        # running SWAG
        params = common_params.copy()
        params["step_sizes"] = step_sizes["swag"]
        params["algorithm"] = "swag"
        barrier(100)
        res = launch(template_name="swag", **params)
        console.print("Running SWAG for all step sizes")

        # running Monte Carlo dropout
        for dropout in dropouts:
            params = common_params.copy()
            params["step_sizes"] = step_sizes["mcdropout"]
            params["dropout"] = dropout
            params["algorithm"] = "mcdropout"
            barrier(100)
            res = launch(template_name="mcdropout", **params)
            console.print(
                f"Running MC dropout with rate {dropout} and for all step sizes"
            )

        # running SGMCMC and deep ensembles
        barrier(100)
        step_sizes_ = step_sizes.copy()
        # step_sizes_.pop("mcdropout")
        # step_sizes_.pop("swag")
        # step_sizes_.pop("laplace")
        for key, val in step_sizes_.items():
            for step_size in val:
                params = common_params.copy()
                params["step_size"] = step_size
                params["algorithm"] = key
                params["init_method"] = "map"
                barrier(100)
                res = launch(template_name="bayesian", **params)
                console.print(f"Running {key} with step size {step_size}")

        # running hmc
        params = common_params.copy()
        params["algorithm"] = "hmc"
        barrier(100)
        res = launch(template_name="bayesian", **params)
        console.print("Running HMC")

    barrier(1)
    clean(args.clean_slurm_outputs)
    for seed in range(NUM_SEEDS):
        common_params["seed"] = seed

        # lengthscale estimation
        res = launch(template_name="lengthscale", **common_params)
        console.print("Running lengthscale computation")

    barrier(1)
    clean(args.clean_slurm_outputs)
    for seed in range(NUM_SEEDS):
        common_params["seed"] = seed

        # ksd computations
        step_sizes_ = step_sizes.copy()
        step_sizes_.pop("mcdropout")
        params = common_params.copy()
        params["algorithms"] = step_sizes_.keys()
        res = launch(template_name="kernelized_stein_discrepancies", **params)
        console.print("Running KSD computation")

    barrier(1)
    clean(args.clean_slurm_outputs)
    for seed in range(NUM_SEEDS):
        common_params["seed"] = seed

        # mmd computations
        res = launch(template_name="maximum_mean_discrepancies", **common_params)
        console.print("Running MMD computation.")

    barrier(1)
    clean(args.clean_slurm_outputs)
    for seed in range(NUM_SEEDS):
        common_params["seed"] = seed

        # coverage probabilities and other metrics
        params = common_params.copy()
        params["algorithms"] = step_sizes.keys()
        res = launch(template_name="metrics_prediction_intervals", **params)
        console.print("Running metrics computation...")

    barrier(1)
    clean(args.clean_slurm_outputs)

    print(f"Removing all the heavy data in {args.workdir}")
    shutil.rmtree(args.workdir)
