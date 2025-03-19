## Introduction

This repository gathers the algorithms and numerical experiments presented in the benchmark paper 
[Benchmarking Bayesian neural networks and evaluation metrics for regression tasks](https://arxiv.org/abs/2206.06779).

Please consider citing our paper if you find this code useful in your work:
```bibtex
article{staber2023benchmarking,
      title={Benchmarking Bayesian neural networks and evaluation metrics for regression tasks}, 
      author={Brian Staber and Sébastien Da Veiga},
      year={2023},
      eprint={2206.06779},
      archivePrefix={arXiv},
}
```

**This repository is a fork of [`pbnn`](https://gitlab.com/drti/pbnn/) from GitLab. It has been migrated to GitHub for further development.**

This project was originally developed on GitLab. The original repository remains available at [GitLab](https://gitlab.com/drti/pbnn/).

It is licensed under the MIT License – see the [LICENSE](LICENSE.txt) file for details.

## Getting started

### Install guide

You can install this package using `pip`:

```bash
pip install pbnn
```

Note that `pbnn` relies on [JAX](https://github.com/google/jax) which will be installed through [BlackJAX](https://github.com/blackjax-devs/blackjax), the main dependance of this package. 
The code will run on CPU only unless you install JAX with GPU support (see [official instructions](https://github.com/google/jax#installation)). 
JAX has been mainly chosen for its composable function transformations (such as `grad`, `jit`, or `scan`) that make MCMC methods for 
neural networks computationally tractable.

By relying on [Flax](https://github.com/google/flax) and [BlackJAX](https://github.com/blackjax-devs/blackjax), `pbnn` gives acess to (SG)MCMC methods (most of them being simplified user interfaces built on top of BlackJAX), but also to deep ensembles, Monte Carlo dropout, stochastic weight averaging Gaussian (SWAG), and classical MAP estimation.

The remaining algorithms tested in the accompanying [paper](https://arxiv.org/abs/2206.06779) are taken from [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) for conformal prediction, and [laplace](https://github.com/aleximmer/Laplace) for effective Laplace approximation in [PyTorch](https://github.com/pytorch/pytorch). As such, these are not accessible via `pbnn`, and the interested reader is referred to the `benchmark` folder, or the packages official documentations.

### Documentation

API documentation and several examples of usage can be found in the [online documentation](https://pbnn.readthedocs.io/en/latest/?pbnn=latest).