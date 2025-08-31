"""Utility functions for data handling."""
# # This file is subject to the terms and conditions defined in
# # file 'LICENSE.txt', which is part of this source code package.
#

import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data
from jax.tree_util import tree_map


def batch_unlabeled_data(rng_key, data, batch_size, data_size, replace=True):
    """Batch generator for unlabeled data.

    Parameters
    ----------

    rng_key
        Random seed key
    data
        JAX Array
    batch_size
        batch_size
    data_size
        Dataset size
    replace
        Sample with replacement (default: True)

    Returns:
    -------
    Generator yielding batches of data

    """
    while True:
        _, rng_key = jax.random.split(rng_key)
        idx = jax.random.choice(
            key=rng_key,
            a=jnp.arange(data_size),
            shape=(batch_size,),
            replace=replace,
        )
        minibatch = data[idx]
        yield minibatch


def batch_labeled_data(rng_key, data, batch_size, data_size, replace=True):
    """Batch generator for labeled data.

    Parameters
    ----------

    rng_key
        Random seed key
    data
        Tuple (X, y) of labeled data
    batch_size
        batch_size
    data_size
        Dataset size
    replace
        Sample with replacement (default: True)

    Returns:
    -------
    Generator yielding batches of data

    """
    while True:
        _, rng_key = jax.random.split(rng_key)
        idx = jax.random.choice(
            key=rng_key,
            a=jnp.arange(data_size),
            shape=(batch_size,),
            replace=replace,
        )
        minibatch = tuple(elem[idx] for elem in data)
        yield minibatch


def numpy_collate(batch):
    """Collate function for numpy arrays."""
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyDataset(data.Dataset):
    """Dataset for numpy arrays."""

    def __init__(self, X, Y):
        """Dataset for numpy arrays."""
        self.X = np.array(X)
        self.Y = np.array(Y)

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.X)

    def __getitem__(self, index):
        """Return a single data point."""
        return (self.X[index], self.Y[index])


class NumpyLoader(data.DataLoader):
    """DataLoader for numpy arrays."""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        """DataLoader for numpy arrays.

        Args:
            dataset: Dataset from which to load the data.
            batch_size: How many samples per batch to load (default: 1).
            shuffle: Set to True to have the data reshuffled at every epoch (default: False).
            sampler: Defines the strategy to draw samples from the dataset. If specified,
                     shuffle must be False.
            batch_sampler: Like sampler, but returns a batch of indices at a time. Mutually
                           exclusive with batch_size, shuffle, sampler, and drop_last.
            num_workers: How many subprocesses to use for data loading. 0 means that the
                         data will be loaded in the main process (default: 0).
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
                        before returning them (default: False).
            drop_last: Set to True to drop the last incomplete batch.
            timeout: The timeout value for collecting a batch from workers. Should always be
                     non-negative (default: 0).
            worker_init_fn: If not None, this will be called on each worker subprocess with
                            the worker id (an int in [0, num_workers - 1]) as input, after
                            seeding and before data loading. (default: None).
        """
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
