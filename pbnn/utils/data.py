import jax
import jax.numpy as jnp


def batch_unlabeled_data(rng_key, data, batch_size, data_size, replace=True):
    """Return an iterator over batches of data."""
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
    """Return an iterator over batches of data."""
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
