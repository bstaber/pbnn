import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pbnn.utils.analytical_functions import (
    trigonometric_function,
    heteroscedastic_trigonometric_function,
    ishigami_function,
    gramacy_function,
    g_function,
)


def plot(x, y, ls="", color="k", marker="o", xlabel=r"$x$"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, ls=ls, color=color, marker=marker)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    return fig


# test function 1
noise_level = 0.0
x = np.linspace(-3, 3, 200)[:, None]
noise = noise_level * np.random.randn(*x.shape)
y = trigonometric_function(x, noise)
fig = plot(x, y)
fig.savefig("trigonometric_function.png", format="png")

# test function 2
noise_level = 1.0
x = np.random.uniform(low=0, high=1.0, size=(20000, 1000))
noise = noise_level * np.random.randn(x.shape[0], 1)
y = heteroscedastic_trigonometric_function(x, noise)
beta = np.zeros((x.shape[1],))
beta[0:5] = 1.0
fig = plot(np.dot(x, beta), y)
fig.savefig("heteroscedastic_trigonometric_function.png", format="png")

# test function 3
noise_level = 0.0
x = np.random.uniform(low=-np.pi, high=np.pi, size=(200, 3))
noise = noise_level * np.random.randn(len(x))
y = ishigami_function(x, noise)
fig = plt.figure(constrained_layout=True, figsize=(3 * 5, 5))
gs = GridSpec(nrows=1, ncols=3, figure=fig)
for i in range(3):
    ax = fig.add_subplot(gs[i])
    ax.plot(x[:, i], y, ls="", color="k", marker="o")
    ax.set_xlabel(rf"$x_{i}$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
fig.savefig("ishigami_function.png", format="png")

# test function 4
noise_level = 0.0
x = np.linspace(0, 20, 200)[:, None]
noise = noise_level * np.random.randn(len(x), 1)
y = gramacy_function(x, noise)
fig = plot(x, y)
fig.savefig("gramacy_function.png", format="png")

# test function 5
noise_level = 0.0
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
x = np.stack([X.ravel(), Y.ravel()], axis=1)
noise = noise_level * np.random.randn(len(x))
y = g_function(x, noise)
fig = plt.figure(constrained_layout=True, figsize=(2 * 5, 5))
gs = GridSpec(nrows=1, ncols=2, figure=fig)
for i in range(2):
    ax = fig.add_subplot(gs[i])
    ax.plot(x[:, i], y, ls="", color="k", marker="o")
    ax.set_xlabel(rf"$x_{i}$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
fig.savefig("g_function.png", format="png")
