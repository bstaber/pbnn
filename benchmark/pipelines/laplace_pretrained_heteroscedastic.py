from pathlib import Path
from time import time

import numpy as np
import torch
from absl import app, flags
from experiments import load_experiment
from hetreg.marglik import marglik_optimization
from laplace import KronLaplace

# from laplace import Laplace
# from laplace.curvature.backpack import BackPackGGN
from laplace.curvature.asdl import AsdlGGN
from rich.progress import track
from torch.utils.data import DataLoader, TensorDataset

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
flags.DEFINE_float("step_size", 1e-3, "Step size of the algorithm")
flags.DEFINE_integer(
    "seed", default=0, help="Initial seed that will be split accross the functions"
)


def setup_directories(workdir, seed, experiment, step_size):
    WORKDIR = Path(workdir)
    DATADIR = WORKDIR / f"seed_{seed}" / experiment.name / "laplace" / f"lr_{step_size}"
    DATADIR.mkdir(parents=True, exist_ok=True)
    return DATADIR


# def train_model(model, train_loader, optimizer, loss_fn, num_epochs=1000):
#     for _ in range(num_epochs):
#         for x_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             pred = model(x_batch)
#             loss = loss_fn(y_batch, pred)
#             loss.backward()
#             optimizer.step()


def main(argv):
    """Main function to train a model, fit a Laplace approximation, and save results.

    Args:
        argv (list): Command line arguments.
    """
    workdir = FLAGS.workdir
    step_size = FLAGS.step_size
    num_datasets = FLAGS.num_datasets
    seed = FLAGS.seed

    experiment = load_experiment(index=FLAGS.experiment)

    hessian_structure = "kron"
    sigma_noise = experiment.noise_level
    prior_precision = 1.0

    DATADIR = setup_directories(workdir, seed, experiment, step_size)

    for i in track(range(num_datasets), total=num_datasets):
        X, _, y, X_test, _, _ = experiment.load_data_fn(i)

        X, y = (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
        )

        model = experiment.network_torch()

        train_loader = DataLoader(
            dataset=TensorDataset(X, y), batch_size=32, shuffle=True, drop_last=True
        )

        t_initial = time()

        lr = step_size
        lr_min = 1e-5
        lr_hyp = 1e-1
        lr_hyp_min = 1e-1
        marglik_early_stopping = True
        n_epochs = 10000
        n_hypersteps = 50
        marglik_frequency = 50
        laplace = KronLaplace
        optimizer = "Adam"
        backend = AsdlGGN
        n_epochs_burnin = 100
        prior_prec_init = 1e-3

        la, model, margliksh, _, _ = marglik_optimization(
            model,
            train_loader,
            likelihood="heteroscedastic_regression",
            lr=lr,
            lr_min=lr_min,
            lr_hyp=lr_hyp,
            early_stopping=marglik_early_stopping,
            lr_hyp_min=lr_hyp_min,
            n_epochs=n_epochs,
            n_hypersteps=n_hypersteps,
            marglik_frequency=marglik_frequency,
            laplace=laplace,
            prior_structure="layerwise",
            backend=backend,
            n_epochs_burnin=n_epochs_burnin,
            scheduler="cos",
            optimizer=optimizer,
            prior_prec_init=prior_prec_init,
            use_wandb=False,
        )

        t_final = time()

        X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
        f_mu, f_var, y_var = la(X_test)

        positions = la.sample(n_samples=2000).detach().cpu().numpy()

        predictions = la.predictive_samples(X_test, n_samples=2000)

        np.savez(
            # os.path.join(root, f"saved_data_{i}.npz"),
            DATADIR / f"saved_data_{i}.npz",
            positions=positions,
            predictions=predictions.detach().cpu().numpy(),
            f_mu=f_mu.detach().cpu().numpy(),
            f_var=f_var.detach().cpu().numpy(),
            y_var=y_var.detach().cpu().numpy(),
            time=t_final - t_initial,
        )


if __name__ == "__main__":
    app.run(main)
