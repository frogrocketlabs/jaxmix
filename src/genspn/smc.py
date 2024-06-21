import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from jaxtyping import Array, Float, Int
from genspn.distributions import (NormalInverseGamma, Dirichlet, MixedConjugate, 
    posterior, sample, logpdf, Normal, Categorical, Mixed, GEM, Cluster, Trace)
from functools import partial


def step(data, gibbs_iters, key, K, trace, max_clusters):
    q_split_trace = q_split(data, gibbs_iters, max_clusters, key, trace)

    weights = get_weights(trace, q_split_trace, max_clusters)
    k = jnp.random.categorical(key, weights)

    if k == 0:
        return trace
    else:
        return split_cluster(trace, k-1)

def get_weights(trace, K, data, q_split_trace, max_clusters):
    # for each cluster, get the pi score
    # I think the idea will be to make a separate pi vector for each 
    # possible cluster split
    pi_split = jax.vmap(make_pi, in_axes=(None, 0, None, None))(
        trace.pi, k, q_split_trace.pi, max_clusters)
    logpdf_pi = jax.vmap(logpdf, in_axes=(None, 0, None))(trace.GEM, pi_split, K)

    logpdf_clusters = score_trace_cluster(data, trace.g, trace.cluster, trace)
    logpdf_except_cluster = jnp.sum(logpdf_clusters) - logpdf_clusters

    # score pi1 and pi2 under each proposal
    q_pi_dist = Dirichlet(alpha=jnp.ones(2) * trace.gem.alpha/2)
    q_pi = jnp.vstack((
        q_split_trace.cluster.pi[jnp.arange(K)],
        q_split_trace.cluster.pi[max_clusters + jnp.arange(K)],
    ))

    logpdf_q_pi = jax.vmap(logpdf, in_axes=(None, 0))(q_pi_dist, q_pi)

    return logpdf_pi + logpdf_except_cluster - logpdf_q_pi

def score_trace_cluster(data, g, cluster, max_clusters):
    c, pi, f = cluster.c, cluster.pi, cluster.f

    x_scores = jax.vmap(logpdf, in_axes=(None, 0, 0))(f, data, c)
    pi_dist = Categorical(logprobs=jnp.log(pi.reshape(1, -1)))
    c_scores = jax.vmap(logpdf, in_axes=(None, 0))(pi_dist, c.reshape(-1, 1))
    theta_scores = jax.vmap(logpdf, in_axes=(None, 0))(g, f)

    xc_scores_cluster = jax.ops.segment_sum(
        x_scores + c_scores, c, 
        num_segments=max_clusters)

    return xc_scores_cluster + theta_scores

def split_cluster(trace, k):
    pass

@partial(jax.jit, static_argnames=['data', 'gibbs_iters', 'max_clusters'])
def q_split(data, gibbs_iters, max_clusters, key, trace) -> Cluster:
    c0 = trace.cluster.c
    keys = jax.random.split(key, 3)
    c = (c0 + max_clusters * jax.random.bernoulli(keys[0], shape=c0.shape)).astype(int)

    log_likelihood_mask = make_log_likelihood_mask(c, max_clusters)

    partial_gibbs_step = partial(gibbs_step, 
        alpha=trace.gem.alpha, g=trace.g, data=data, 
        log_likelihood_mask=log_likelihood_mask, max_clusters=max_clusters)

    keys = jax.random.split(keys[2], gibbs_iters)
    _, q_split_trace = jax.lax.scan(partial_gibbs_step, c, keys)

    return q_split_trace

def make_log_likelihood_mask(c, max_clusters):
    log_likelihood_mask = -jnp.inf * jnp.ones((c.shape[0], 2 * np.array(max_clusters, dtype=int)))

    n = jnp.arange(c.shape[0], dtype=int)
    clusters_x = jnp.concatenate((n, n))
    clusters_y = jnp.concatenate((c, max_clusters + c), dtype=int)
    return log_likelihood_mask.at[clusters_x, clusters_y].set(0)

def gibbs_step(c, key, alpha, g, data, log_likelihood_mask, max_clusters):
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    f = gibbs_f(max_clusters, data, subkey1, g, c)
    pi = gibbs_pi(max_clusters, subkey2, alpha, c)
    c = gibbs_c(subkey3, pi, log_likelihood_mask, f, data)

    # now compute the logpdf for the assignments given the new distribution
    return c, Cluster(c, pi, f)

def gibbs_f(max_clusters, data, key, g, c):
    g_prime = posterior(g, data, c, 2*max_clusters)
    return sample(key, g_prime)

def gibbs_c(key, pi, log_likelihood_mask, f, data):
    log_likelihoods = jax.vmap(jax.vmap(logpdf, in_axes=(0, None)), in_axes=(None, 0))(f, data)
    log_likelihoods = log_likelihoods + log_likelihood_mask
    log_score = log_likelihoods + jnp.log(pi)

    return jax.random.categorical(key, log_score, axis=-1).astype(int)

def gibbs_pi(max_clusters, key, alpha, c):
    cluster_counts = jnp.sum(jax.nn.one_hot(c, num_classes=2 * max_clusters, dtype=jnp.int32), axis=0)
    pi = jax.random.dirichlet(key, alpha / 2 + cluster_counts)
    pi_pairs = pi.reshape((2, -1))
    pi_pairs = pi_pairs / jnp.sum(pi_pairs, axis=0)
    return pi_pairs.reshape(-1)

def score_clusters(c, pi, theta, x, max_clusters):
    pass