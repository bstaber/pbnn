import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_all_results_with_hparam_names(base_path, seed, test_case, algorithm):
    """
    Automatically load all results for a given algorithm and seed, storing the data in a dict
    where keys are dicts containing hyperparameters names and values, and the values are the loaded data.

    Parameters:
    - base_path: str or Path, the root directory where results are stored.
    - seed: int, the seed number.
    - test_case: str, the name of the test case.
    - algorithm: str, the algorithm name.

    Returns:
    - results_dict: dict, where the keys are dicts of hyperparameters and the values are the loaded data.
    """
    # Build the base path
    base_dir = Path(base_path) / f"seed_{seed}" / test_case / algorithm

    # Glob all result paths
    results_paths = list(base_dir.glob("**/metrics_pi.npz"))

    if not results_paths:
        print(f"No results found for {algorithm} in {base_dir}")
        return {}

    results_dict = {}

    # Regex patterns for extracting hyperparameters
    lr_pattern = re.compile(r"lr_([0-9\.eE\-]+)")
    rate_pattern = re.compile(r"rate_([0-9\.]+)")

    for result_path in results_paths:
        # Extract learning rate (lr)
        lr_match = lr_pattern.search(str(result_path))
        lr = float(lr_match.group(1)) if lr_match else None

        # Prepare a dictionary for the hyperparameters
        hparams = {"lr": lr}

        # Extract dropout rate (only for mcdropout)
        if algorithm == "mcdropout":
            rate_match = rate_pattern.search(str(result_path))
            rate = float(rate_match.group(1)) if rate_match else None
            hparams["rate"] = rate

        # Load the .npz file
        try:
            data = np.load(result_path)
            results_dict[frozenset(hparams.items())] = data
        except Exception as e:
            print(f"Error loading file {result_path}: {e}")

    return results_dict


def aggregate_metrics_with_shapes(base_path, test_case, algorithm):
    """
    Aggregates metrics across seeds and handles shape mismatches by padding shorter arrays.

    Parameters:
    - base_path: str, base directory containing results.
    - test_case: str, test case name.
    - algorithm: str, algorithm name.

    Returns:
    - dict: Aggregated metrics with mean and std for each hyperparameter combination.
    """
    import os
    from glob import glob

    import numpy as np

    aggregated_results = defaultdict(lambda: defaultdict(list))

    # Glob all results for the given algorithm
    results_paths = glob(
        os.path.join(base_path, f"seed_*/{test_case}/{algorithm}/*/metrics_pi.npz")
    )

    # Extract metrics
    for path in results_paths:
        hparams = frozenset(
            tuple(kv.split("_")) for kv in path.split("/")[-2].split("/")
        )
        data = np.load(path)
        for metric_name in data.files:
            metric_array = data[metric_name]
            aggregated_results[hparams][metric_name].append(metric_array)

    # Aggregate across seeds
    final_results = {}
    for hparams, metrics in aggregated_results.items():
        final_metrics = {}
        for metric_name, metric_values in metrics.items():
            # Determine max shape along the first axis
            max_length = max(arr.shape[0] for arr in metric_values)
            # Pad arrays to the maximum length
            padded_values = [
                np.pad(
                    arr,
                    [(0, max_length - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1),
                    mode="constant",
                    constant_values=np.nan,  # Use NaN to indicate missing data
                )
                for arr in metric_values
            ]
            # Stack padded arrays
            stacked_values = np.stack(padded_values)
            # Compute mean and std, ignoring NaNs
            final_metrics[metric_name] = {
                "mean": np.nanmean(stacked_values, axis=0),
                "std": np.nanstd(stacked_values, axis=0),
            }
        final_results[hparams] = final_metrics

    return final_results


def plot_metrics(aggregated_metrics, algorithm_name):
    lr_values, mcp_means, mcp_stds = [], [], []
    mae_ccp_means, mae_ccp_stds, q2_means, q2_stds = [], [], [], []

    for hparams, metrics in aggregated_metrics.items():
        lr = float(dict(hparams).get("lr", 0))
        lr_values.append(lr)
        mcp_means.append(metrics["mcp"]["mean"][-1])  # Last confidence level
        mcp_stds.append(metrics["mcp"]["std"][-1])
        mae_ccp_means.append(metrics["mae_ccp"]["mean"][-1])
        mae_ccp_stds.append(metrics["mae_ccp"]["std"][-1])
        q2_means.append(metrics["q2_scores"]["mean"])
        q2_stds.append(metrics["q2_scores"]["std"])

    lr_values, mcp_means, mcp_stds = map(np.array, [lr_values, mcp_means, mcp_stds])
    mae_ccp_means, mae_ccp_stds, q2_means, q2_stds = map(
        np.array, [mae_ccp_means, mae_ccp_stds, q2_means, q2_stds]
    )

    # Sort by learning rate for better visualization
    sort_indices = np.argsort(lr_values)
    lr_values = lr_values[sort_indices]

    mcp_means, mcp_stds = mcp_means[sort_indices], mcp_stds[sort_indices]
    mae_ccp_means, mae_ccp_stds = (
        mae_ccp_means[sort_indices],
        mae_ccp_stds[sort_indices],
    )
    q2_means, q2_stds = q2_means[sort_indices], q2_stds[sort_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lr_values,
            y=mcp_means,
            mode="lines+markers",
            error_y=dict(type="data", array=mcp_stds, visible=True),
            name="MCP",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lr_values,
            y=mae_ccp_means,
            mode="lines+markers",
            error_y=dict(type="data", array=mae_ccp_stds, visible=True),
            name="MAE CCP",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lr_values,
            y=q2_means,
            mode="lines+markers",
            error_y=dict(type="data", array=q2_stds, visible=True),
            name="Q2 Scores",
        )
    )

    fig.update_layout(
        title=f"Metrics for {algorithm_name}",
        xaxis_title="Learning Rate",
        yaxis_title="Metric Value",
        legend_title="Metrics",
    )
    return fig


if __name__ == "__main__":
    # Example usage:
    base_path = "results"
    seed = 0
    test_case = "trigonometric_function"
    algorithm = "sgld"  # or any other algorithm, e.g., "sgld"

    results_dict = load_all_results_with_hparam_names(
        base_path, seed, test_case, algorithm
    )

    # Access the results for specific hyperparameters
    # The keys are frozensets of hyperparameter name-value pairs
    for hparams, data in results_dict.items():
        print(
            f"Hyperparameters: {dict(hparams)}"
        )  # Convert frozenset back to dict for readability
        print(f"Metrics: {np.mean(data['q2_scores'])}")  # Replace 'metrics' with the actual key in your npz file
        # print(data["mcp"])

    # # Example usage
    # aggregated_results = aggregate_metrics_with_shapes(
    #     base_path, "trigonometric_function", "sgld"
    # )

    # # Print aggregated results for a hyperparameter combination
    # for hparams_frozen, metrics in aggregated_results.items():
    #     hparams = dict(hparams_frozen)
    #     print(f"Hyperparameters: {hparams}")
    #     for metric_name, stats in metrics.items():
    #         print(f"Metric: {metric_name}")
    #         print(f"Mean Shape: {stats['mean'].shape}, Std Shape: {stats['std'].shape}")
    #         print("---")

    # fig = plot_metrics(aggregated_results, "sgld")
    # fig.write_image("test.png")
