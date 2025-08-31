# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#


import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array


def plot_interval(
    X: Array,
    X_test: Array,
    y: Array,
    y_test: Array,
    y_pred: Array,
    alpha: float = 0.05,
    save_to_file=None,
    use_tex: bool = False,
):
    """Function that plots a prediction and the associated interval along with
    the training and test data. Only works for one-dimensional problems.

    Parameters
    ----------

    X
        Vector of training input features of size (N, )
    X_test
        Vector of test input features of size (M, )
    y
        Vector of training outputs of size (N, 1)
    y_test
        Vector of test outputs of size (M, 1)
    y_pred
        Vector of predictions on the test set of size (M, )
    alpha
        Confidence level (default: 0.05)
    save_to_file
        Filename (without extension) to be used for saving the picture (default: None, not saving the figure)
    use_tex
        Use LaTeX for rendering

    Returns:
    -------
    Figure handle.

    """
    mpl_options = {
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "text.usetex": True,
    }
    plt.rcParams.update(mpl_options) if use_tex else False

    X_test = X_test.squeeze()

    mean_prediction = jnp.median(y_pred, axis=0)
    qlow = jnp.quantile(y_pred, 0.5 * alpha, axis=0)
    qhigh = jnp.quantile(y_pred, (1 - 0.5 * alpha), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_on_axis(ax, X_test, y_test, mean_prediction, qlow, qhigh)

    if save_to_file is not None:
        fig.savefig(f"{save_to_file}.pdf", format="pdf")

    return fig


def plot_on_axis(ax, X_test, y_test, mean_prediction, qlow, qhigh, title=None):
    # interval
    ax.fill_between(X_test.squeeze(), qlow, qhigh, color="tab:red", alpha=0.5)
    ax.plot(
        X_test,
        y_test,
        ls="",
        marker="o",
        markerfacecolor="b",
        markeredgecolor="k",
        markeredgewidth=1.0,
        alpha=0.9,
        label="Test data",
    )
    ax.plot(X_test, qlow, ls="--", color="tab:red", label=r"$q_{0.025}$")
    ax.plot(X_test, qhigh, ls="--", color="tab:red", label=r"$q_{0.975}$")
    # prediction
    ax.plot(X_test, mean_prediction, ls="-", color="tab:red", lw=2, label="Mean pred.")
    ax.legend(fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=14)
    return ax
