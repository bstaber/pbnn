import jax
import jax.numpy as jnp


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
        
    Returns
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
        
    Returns
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
