## Introduction

This repository gathers the algorithms and numerical experiments presented in the benchmark paper 
[Benchmarking Bayesian neural networks and evaluation metrics for regression tasks](https://arxiv.org/abs/2206.06779). 

Please consider citing our paper if you find this code useful in your work:
article{staber2023benchmarking,
      title={Benchmarking Bayesian neural networks and evaluation metrics for regression tasks}, 
      author={Brian Staber and Sébastien Da Veiga},
      year={2023},
      eprint={2206.06779},
      archivePrefix={arXiv},
}

## Getting started

### Install guide

You can install this package using `pip`:

```bash
pip install pbnn
```

Note that `pbnn` relies on JAX which will be installed through BlackJAX, the main dependance of this package. 
The code will run on CPU only unless you install JAX with GPU support (see [officiel instructions](https://github.com/google/jax#installation)).

### Documentation

API documentation and several examples of usage can be found in the [online documentation](https://pbnn.readthedocs.io/en/latest/?pbnn=latest).