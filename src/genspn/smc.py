import jax.numpy as jnp
import jax
import numpy as np
import time
from genspn.distributions import NormalInverseGamma, Dirichlet, MixedConjugate, posterior, sample, logpdf
from functools import partial


@partial(jax.jit, static_argnames=['gibbs_iters'])
def step(alpha, h, data, key, c, gibbs_iters):
    max_clusters = 50
    keys = jax.random.split(key, 3)
    assignments = c + max_clusters * jax.random.bernoulli(keys[0], shape=c.shape)
    assignments = jnp.array(assignments, dtype=int)

    log_likelihood_mask = -jnp.inf * jnp.ones((assignments.shape[0], 2 * np.array(max_clusters, dtype=int)))

    n = jnp.arange(c.shape[0], dtype=int)
    clusters_x = jnp.concatenate((n, n))
    clusters_y = jnp.concatenate((c, max_clusters + c), dtype=int)
    log_likelihood_mask = log_likelihood_mask.at[clusters_x, clusters_y].set(0)

    partial_gibbs_step = partial(gibbs_step, alpha=alpha, h=h, data=data, log_likelihood_mask=log_likelihood_mask, max_clusters=max_clusters)

    keys = jax.random.split(keys[2], gibbs_iters)
    _, trace = jax.lax.scan(partial_gibbs_step, assignments, keys)

    assignments, pi, theta = trace

    return assignments, pi, theta

def gibbs_step(assignments, key, alpha, h, data, log_likelihood_mask, max_clusters):
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    h_prime = posterior(h, data, assignments, 2*max_clusters)
    theta = sample(subkey1, h_prime)

    # sample pi from the posterior 
    cluster_counts = jnp.sum(jax.nn.one_hot(assignments, num_classes=2 * max_clusters, dtype=jnp.int32), axis=0)
    pi = jax.random.dirichlet(subkey2, alpha / 2 + cluster_counts)
    pi_pairs = pi.reshape((2, -1))
    pi_pairs = pi_pairs / jnp.sum(pi_pairs, axis=0)
    pi = pi_pairs.reshape(-1)

    # now compute the logpdf for the assignments given the new distribution
    log_likelihoods = jax.vmap(jax.vmap(logpdf, in_axes=(0, None)), in_axes=(None, 0))(theta, data)
    log_likelihoods = log_likelihoods + log_likelihood_mask
    log_score = log_likelihoods + jnp.log(pi)

    assignments = jax.random.categorical(subkey3, log_score, axis=-1)
    assignments = jnp.array(assignments, dtype=int)

    return assignments, (assignments, pi, theta)