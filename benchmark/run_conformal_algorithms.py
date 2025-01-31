import argparse
import json
import sys

import numpy as np
from rich.console import Console

from utils import launch

sys.path.append("pipelines")

step_sizes = np.logspace(-1, -5, 10)[::-1].tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, choices=np.arange(10))
    parser.add_argument("--experiment", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--template_folder", type=str, default="pipelines/templates")
    parser.add_argument("--command", type=str, default="sbatch")

    args = parser.parse_args()

    with open("slurm_params.json", "r") as f:
        slurm_params = json.load(f)
    common_params = slurm_params.copy()
    common_params["experiment"] = args.experiment
    common_params["seed"] = args.seed
    common_params["workdir"] = args.workdir
    common_params["template_folder"] = args.template_folder
    common_params["command"] = args.command

    console = Console(style="green")

    params = common_params.copy()
    params["algorithm"] = "split_cp"
    for step_size in step_sizes:
        params["step_size"] = step_size
        launch(template_name="conformal", **params)
        
    params = common_params.copy()
    params["algorithm"] = "cv_plus"
    for step_size in step_sizes:
        params["step_size"] = step_size
        launch(template_name="conformal", **params)
        
    params = common_params.copy()
    params["algorithm"] = "split_cqr"
    for step_size in step_sizes:
        params["step_size"] = step_size
        launch(template_name="conformal", **params)
