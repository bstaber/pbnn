import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy as np
import scipy.stats as stats
from experiments import Experiment, load_experiment
from tqdm import tqdm


def normal_confidence_interval_bounds(alpha: float):
    # Calculate the quantiles
    lower_bound = stats.norm.ppf(alpha / 2)
    upper_bound = stats.norm.ppf(1 - alpha / 2)

    return lower_bound, upper_bound


def compute_interval_metrics(alphas, data, algorithm, y_test):
    if algorithm == "laplace":
        f_mu, f_var, sigma_noise = data["f_mu"], data["f_var"], data["sigma_noise"]
        pred_std = np.sqrt(f_var + sigma_noise**2)

        inside = []
        widths_per_alpha = []
        for alpha in alphas:
            lb, ub = normal_confidence_interval_bounds(alpha)
            qlow = f_mu + lb * pred_std
            qhigh = f_mu + ub * pred_std

            inside_dim = (y_test - qhigh[:, None] < 0) * (y_test - qlow[:, None] > 0)
            inside.append(inside_dim)
            
            widths_per_alpha.append(np.mean(qhigh - qlow))
    else:
        # get predictions
        f_predictions = data["predictions"]
        y_predictions = f_predictions + noise_level * np.random.randn(*f_predictions.shape)

        # compute coverage probabilities
        inside = []
        widths_per_alpha = []
        for alpha in alphas:
            qlow = jnp.quantile(y_predictions, 0.5 * alpha, axis=0)
            qhigh = jnp.quantile(y_predictions, (1 - 0.5 * alpha), axis=0)

            inside_dim = (y_test - qhigh[:, None] < 0) * (y_test - qlow[:, None] > 0)
            inside.append(inside_dim)

            widths_per_alpha.append(np.mean(qhigh - qlow))
    inside = np.stack(inside, axis=0)

    return inside, widths_per_alpha


def compute_interval_metrics_heteroscedastic(alphas, data, algorithm, y_test):
    if algorithm == "laplace":
        # f_mu, f_var, y_var = la(x)
        # f_mu, f_var, y_var = f_mu.squeeze(), f_var.squeeze(), y_var.squeeze()
        # mh_map, sh_map = f_mu.numpy(), 2 * np.sqrt(y_var.numpy())
        f_mu, f_var, y_var = data["f_mu"], data["f_var"], data["y_var"]
        f_mu, pred_std = f_mu.numpy(), np.sqrt(f_var.numpy() + y_var.numpy())

        inside = []
        widths_per_alpha = []
        for alpha in alphas:
            lb, ub = normal_confidence_interval_bounds(alpha)
            qlow = f_mu + lb * pred_std
            qhigh = f_mu + ub * pred_std

            inside_dim = (y_test - qhigh[:, None] < 0) * (y_test - qlow[:, None] > 0)
            inside.append(inside_dim)
            
            widths_per_alpha.append(np.mean(qhigh - qlow))
    else:
        # get predictions
        f_predictions = data["predictions"]
        y_predictions = f_predictions[:, :, 0] + noise_level * f_predictions[:, :, 1] * np.random.randn(*f_predictions[:, :, 0].shape)

        # compute coverage probabilities
        inside = []
        widths_per_alpha = []
        for alpha in alphas:
            qlow = jnp.quantile(y_predictions, 0.5 * alpha, axis=0)
            qhigh = jnp.quantile(y_predictions, (1 - 0.5 * alpha), axis=0)

            inside_dim = (y_test - qhigh[:, None] < 0) * (y_test - qlow[:, None] > 0)
            inside.append(inside_dim)

            widths_per_alpha.append(np.mean(qhigh - qlow))
    inside = np.stack(inside, axis=0)

    return inside, widths_per_alpha


def compute_regression_metrics(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray)
        y_pred (np.ndarray)

    Returns:
        Metrics: rmse and q2
    """
    rmse = jnp.sqrt(jnp.mean((y_true[:, 0].squeeze() - y_pred.squeeze()) ** 2))
    ssr = jnp.sum((y_pred.squeeze() - y_true[:, 0].squeeze()) ** 2)
    sst = jnp.sum(
        (y_true[:, 0].squeeze() - np.mean(y_true[:, 0], axis=0).squeeze()) ** 2
    )
    q2_score = 1.0 - ssr / sst

    return rmse, q2_score


def compute_coverage_probabilities(alphas: np.ndarray, inside_list: np.ndarray):
    """
    Args:
        alphas (np.ndarray): Array of confidence levels
        inside_list (np.ndarray): Array of bools such that True means that the points falls into the interval. Shape: [#_datasets, #_alphas, #_xtest, #_ytest_per_xtest]

    Returns:
        Train conditional coverage, marginal coverage, mean asbolute error of test conditional coverage
    """
    marginal_coverage = np.zeros(len(alphas))
    mae_test_coverage = np.zeros(len(alphas))
    train_coverage = np.zeros((len(alphas), inside_list.shape[0]))

    for j, alpha in enumerate(alphas):
        train_coverage[j] = np.mean(inside_list[:, j, :, 0], axis=1)
        marginal_coverage[j] = np.mean(inside_list[:, j, :, 0])
        ccp = np.mean(np.mean(inside_list[:, j, :, :], axis=-1), axis=0)
        mae_test_coverage[j] = np.mean(np.abs(ccp - alpha))

    return train_coverage, marginal_coverage, mae_test_coverage


def post_process_data(experiment: Experiment, files: list, algorithm: str):
    """
    Post-processes benchmark data to compute coverage probabilities, RMSE, Q2 scores, and other metrics.

    Parameters:
    - experiment (Experiment): An instance of the Experiment class with a method to load data.
    - files (list): List of file paths containing benchmark data.
    - algorithm (str): The name of the algorithm used for prediction ('laplace' or otherwise).

    Returns:
    - mcp (np.ndarray): Mean coverage probabilities for different alpha levels.
    - mae_ccp (np.ndarray): Mean absolute error of coverage probabilities for different alpha levels.
    - rmse_list (np.ndarray): Root Mean Squared Error for each dataset.
    - q2_scores (np.ndarray): Q2 scores for each dataset.
    - widths (np.ndarray): Average width of confidence intervals for each dataset.
    - times (np.ndarray): Computational time for each dataset.
    """

    inside_list = []
    widths_list = []
    q2_scores = []
    rmse_list = []
    times = []

    for file in files:
        # load data
        data = np.load(file)
        dataset_idx = int(str(file).split("saved_data_")[-1].split(".npz")[0])
        _, _, _, X_test, f_test, y_test = experiment.load_data_fn(dataset_idx)

        # get computational time
        times.append(data["time"])

        if experiment.name == "barber":
            # mean prediction
            if algorithm == "laplace":
                y_pred = data["f_mu"]
            else:
                y_pred = jnp.mean(data["predictions"][:, :, 0], axis=0)

            # compute is_inside arrays
            alphas = np.linspace(0.05, 0.95, 19)
            inside_per_alpha, widths_per_alpha = compute_interval_metrics_heteroscedastic(
                alphas, data, algorithm, y_test
            )
        else:
            # mean prediction
            if algorithm == "laplace":
                y_pred = data["f_mu"]
            else:
                y_pred = jnp.mean(data["predictions"], axis=0)

            # compute is_inside arrays
            alphas = np.linspace(0.05, 0.95, 19)
            inside_per_alpha, widths_per_alpha = compute_interval_metrics(
                alphas, data, algorithm, y_test
            )

        inside_list.append(inside_per_alpha)
        widths_list.append(widths_per_alpha)

        # compute regression metrics
        rmse, q2_score = compute_regression_metrics(y_test, y_pred)

        rmse_list.append(rmse)
        q2_scores.append(q2_score)

    inside_list = np.stack(inside_list, axis=0)

    # deduce marginal and conditional coverages
    train_coverage, marginal_coverage, mae_test_coverage = (
        compute_coverage_probabilities(alphas, inside_list)
    )

    # convert everything to arrays
    widths_list = np.stack(widths_list, axis=0)
    rmse_list = np.array(rmse_list)
    q2_scores = np.array(q2_scores)
    times = np.array(times)

    return (
        train_coverage,
        marginal_coverage,
        mae_test_coverage,
        rmse_list,
        q2_scores,
        widths_list,
        times,
    )


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
    parser.add_argument("--experiment", type=int, choices=np.arange(10))
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "sgld",
            "sgld_cv",
            "sgld_svrg",
            "sghmc",
            "sghmc_cv",
            "sghmc_svrg",
            "pSGLD",
            "cyclical_sgld",
            "deep_ensembles",
            "laplace",
            "swag",
            "mcdropout",
        ],
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the whole experiment"
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    experiment = load_experiment(args.experiment)
    noise_level = experiment.noise_level

    WORKDIR = Path(args.workdir)
    SAVEDIR = Path(args.savedir)
    DATADIR = WORKDIR / f"seed_{args.seed}" / experiment.name / args.algorithm
    SAVEDIR = SAVEDIR / f"seed_{args.seed}" / experiment.name / args.algorithm

    if args.algorithm == "mcdropout":
        dropouts = np.linspace(0.1, 0.9, 9)
        for dropout in dropouts:
            DATADIR_ = DATADIR / f"rate_{dropout}"
            SAVEDIR_ = SAVEDIR / f"rate_{dropout}"

            for leaf in tqdm(sorted(DATADIR_.rglob("lr_*"))):
                filenames = sorted(leaf.glob("saved_data*"))
                learning_rate = leaf.name
                if len(filenames) > 0:
                    (
                        train_coverage,
                        mcp,
                        mae_ccp,
                        rmse_list,
                        q2_scores,
                        widths,
                        times,
                    ) = post_process_data(experiment, filenames, args.algorithm)

                    TARGETDIR = SAVEDIR_ / learning_rate
                    TARGETDIR.mkdir(parents=True, exist_ok=True)

                    jnp.savez(
                        TARGETDIR / "metrics_pi.npz",
                        train_coverage=train_coverage,
                        mcp=mcp,
                        mae_ccp=mae_ccp,
                        rmse_list=rmse_list,
                        q2_scores=q2_scores,
                        widths=widths,
                        times=times,
                    )

    else:
        for leaf in tqdm(sorted(DATADIR.rglob("lr_*"))):
            filenames = sorted(leaf.glob("saved_data*"))
            learning_rate = leaf.name
            if len(filenames) > 0:
                train_coverage, mcp, mae_ccp, rmse_list, q2_scores, widths, times = (
                    post_process_data(experiment, filenames, args.algorithm)
                )

                TARGETDIR = SAVEDIR / learning_rate
                TARGETDIR.mkdir(parents=True, exist_ok=True)

                jnp.savez(
                    TARGETDIR / "metrics_pi.npz",
                    train_coverage=train_coverage,
                    mcp=mcp,
                    mae_ccp=mae_ccp,
                    rmse_list=rmse_list,
                    q2_scores=q2_scores,
                    widths=widths,
                    times=times,
                )
