from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from absl import app, flags
from experiments import load_experiment
from laplace import Laplace
from laplace.curvature.backpack import BackPackGGN
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


def train_model(model, train_loader, optimizer, loss_fn, num_epochs=1000):
    for _ in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(y_batch, pred)
            loss.backward()
            optimizer.step()


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

        optimizer = optim.AdamW(model.parameters(), lr=step_size)
        loss_fn = nn.MSELoss()

        t_initial = time()
        model.train()
        train_model(model, train_loader, optimizer, loss_fn)

        laplace_model = Laplace(
            model=model,
            likelihood="regression",
            subset_of_weights="all",
            hessian_structure=hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision,
            backend=BackPackGGN,
        )
        laplace_model.fit(train_loader)

        # post-hoc hyperparameter tuning
        log_prior = torch.ones(1, requires_grad=True)

        hyper_optimizer = torch.optim.Adam([log_prior], lr=1e-1)
        for _ in range(1000):
            hyper_optimizer.zero_grad()
            neg_marglik = -laplace_model.log_marginal_likelihood(
                log_prior.exp(), sigma_noise
            )
            neg_marglik.backward()
            hyper_optimizer.step()
        t_final = time()

        X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
        f_mu, f_var = laplace_model(X_test)
        f_mu, f_var = f_mu.squeeze(), f_var.squeeze()

        positions = laplace_model.sample(n_samples=2000).detach().cpu().numpy()

        # f_mu = f_mu.squeeze().detach().cpu().numpy()
        # f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        # pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        # predictions = (
        #     f_mu[None]
        #     + np.sqrt(f_var) * np.random.randn(2000, len(f_mu))
        #     + experiment.noise_level * np.random.randn(2000, len(f_mu))
        # )

        predictions = laplace_model.predictive_samples(X_test, n_samples=2000)

        np.savez(
            # os.path.join(root, f"saved_data_{i}.npz"),
            DATADIR / f"saved_data_{i}.npz",
            positions=positions,
            predictions=predictions.detach().cpu().numpy(),
            f_mu=f_mu.detach().cpu().numpy(),
            f_var=f_var.detach().cpu().numpy(),
            sigma_noise=laplace_model.sigma_noise.item(),
            time=t_final - t_initial,
        )


if __name__ == "__main__":
    app.run(main)
