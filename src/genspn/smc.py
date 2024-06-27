import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from plum import dispatch
from jaxtyping import Array, Float, Int
from genspn.distributions import (NormalInverseGamma, Dirichlet, MixedConjugate, 
    posterior, sample, logpdf, Normal, Categorical, Mixed, GEM, Cluster, Trace)
from functools import partial

# def smc(trace, max_K):
#     for i in range(2):
#         # sample random assignments to begin with
#         # note: we should flush the cs so they're 1:N
#         new_cluster = step(data, trace=trace, gibbs_iters=iters, max_clusters=3, key=keys[6 + i], K=i+2)
#         trace = Trace(
#             gem=gem,
#             g=g,
#             cluster=new_cluster
#         )

@partial(jax.jit, static_argnames=['gibbs_iters', 'max_clusters'])
def step(data, gibbs_iters, key, K, trace, max_clusters):
    q_split_trace = q_split(data, gibbs_iters, max_clusters, key, trace.cluster.c, trace.gem.alpha, trace.g)

    cluster_weights = get_weights(trace, K, data, q_split_trace, max_clusters)

    logpdf_pi = logpdf(trace.gem, jnp.sort(trace.cluster.pi, descending=True), K-1)
    logpdf_clusters = score_trace_cluster(data, trace.g, trace.cluster, max_clusters)

    idx = jnp.arange(logpdf_clusters.shape[0])
    logpdf_clusters = jnp.where(idx < K-1, logpdf_clusters, 0) 
    stop_weight = logpdf_pi + jnp.sum(logpdf_clusters)

    weights = jnp.zeros(max_clusters + 1)
    weights = weights.at[1:].set(cluster_weights)
    weights = weights.at[0].set(stop_weight)
    k = jax.random.categorical(key, weights)

    return jax.lax.cond(k==0, 
        lambda cluster0, cluster1, k, K, max_clusters: trace.cluster, 
        lambda  cluster0, cluster1, k, K, max_clusters: split_cluster(cluster0, cluster1, k, K, max_clusters),
        trace.cluster, q_split_trace[-1], k-1, K, max_clusters)

def get_weights(trace, K, data, q_split_trace, max_clusters):
    # for each cluster, get the pi score
    # I think the idea will be to make a separate pi vector for each 
    # possible cluster split
    pi_split = jax.vmap(make_pi, in_axes=(None, 0, None, None))(
        trace.cluster.pi, jnp.arange(max_clusters), q_split_trace.pi[-1], max_clusters)
    logpdf_pi = jax.vmap(logpdf, in_axes=(None, 0, None))(trace.gem, pi_split, K)

    logpdf_clusters = score_trace_cluster(data, trace.g, trace.cluster, max_clusters)
    logpdf_except_cluster = jnp.sum(logpdf_clusters) - logpdf_clusters

    logpdf_q_pi = score_q_pi(q_split_trace.pi[-1], max_clusters, trace.gem.alpha)   

    Z = q_Z(q_split_trace[-1], data, max_clusters)

    # score pi1 and pi2 under each proposal
    return Z + logpdf_pi + logpdf_except_cluster - logpdf_q_pi


def q_Z(q_split_trace, data, max_clusters):
    c, pi, f = q_split_trace.c, q_split_trace.pi, q_split_trace.f
    c_mod = jnp.mod(c, max_clusters)
    x_scores0 = jax.vmap(logpdf, in_axes=(None, 0, 0))(f, data, c_mod)
    x_scores1 = jax.vmap(logpdf, in_axes=(None, 0, 0))(f, data, c_mod + max_clusters)

    pi_dist = Categorical(logprobs=jnp.log(pi.reshape(1, -1)))
    c_scores0 = jax.vmap(logpdf, in_axes=(None, 0))(pi_dist, c_mod.reshape(-1, 1))
    c_scores1 = jax.vmap(logpdf, in_axes=(None, 0))(pi_dist, max_clusters + c_mod.reshape(-1, 1))

    scores0 = x_scores0 + c_scores0
    scores1 = x_scores1 + c_scores1

    scores = jnp.logaddexp(scores0, scores1)
    max_score = jnp.max(scores)
    return jnp.log(jax.ops.segment_sum(
        jnp.exp(scores - max_score), c_mod, 
        num_segments=max_clusters)) + max_score

def score_q_pi(q_pi, max_clusters, alpha):
    q_pi_dist = Dirichlet(alpha=jnp.ones((1, 2)) * alpha/2)
    q_pi_stack = Categorical(
        jnp.vstack((
        jnp.log(q_pi[:max_clusters]),
        jnp.log(q_pi[max_clusters:]),
     ))[None, :])

    return jax.vmap(logpdf, in_axes=(None, -1))(q_pi_dist, q_pi_stack)

def make_pi(pi, k, pi_split, max_clusters):
    pi_k0 = pi[k]
    pi = pi.at[k].set(pi_k0 * pi_split[k])
    pi = pi.at[k + max_clusters].set(pi_k0 * pi_split[k + max_clusters])

    return jnp.sort(pi, descending=True)

def score_trace_cluster(data, g, cluster, max_clusters):
    c, pi, f = cluster.c, cluster.pi, cluster.f

    x_scores = jax.vmap(logpdf, in_axes=(None, 0, 0))(f, data, c)
    pi_dist = Categorical(logprobs=jnp.log(pi.reshape(1, -1)))
    c_scores = jax.vmap(logpdf, in_axes=(None, 0))(pi_dist, c.reshape(-1, 1))
    theta_scores = jax.vmap(logpdf, in_axes=(None, 0))(g, f)[:max_clusters]

    xc_scores_cluster = jax.ops.segment_sum(
        x_scores + c_scores, c, 
        num_segments=max_clusters)
    return xc_scores_cluster + theta_scores

def split_cluster(cluster, split_clusters, k, K, max_clusters):
    # update pi
    pi = cluster.pi
    pi0 = pi[k]
    pi = pi.at[k].set(pi0 * split_clusters.pi[k])    
    pi = pi.at[K - 1].set(pi0 * split_clusters.pi[k + max_clusters])    

    # update c
    c = cluster.c
    c = jnp.where(c == k, split_clusters.c, c)
    c = jnp.where(c == k + max_clusters, K-1, c)

    # update f
    f = update_f(cluster.f, split_clusters.f, k, K-1, max_clusters)

    return Cluster(c, pi, f)

@dispatch
def update_f(f0: Normal, f: Normal, k: Int[Array, ""], K: Int[Array, ""], max_clusters: Int[Array, ""]):
    mu = update_vector(f0.mu, f.mu, k, K, max_clusters)
    std = update_vector(f0.std, f.std, k, K, max_clusters)

    return Normal(mu, std)

@dispatch
def update_f(f0: Categorical, f: Categorical, k: Int[Array, ""], K: Int[Array, ""], max_clusters: Int[Array, ""]):
    logprobs = update_vector(f0.logprobs, f.logprobs, k, K, max_clusters)

    return Categorical(logprobs)

@dispatch
def update_f(f0: Mixed, f: Mixed, k: Int[Array, ""], K: Int[Array, ""], max_clusters: Int[Array, ""]):
    return Mixed(
        update_f(f0.normal, f.normal, k, K, max_clusters),
        update_f(f0.categorical, f.categorical, k, K, max_clusters)
    )

def update_vector(v0, split_v, k, K, max_clusters):
    v = v0
    v = v.at[k].set(split_v[k])
    v = v.at[K].set(split_v[k + max_clusters])
    return v

@partial(jax.jit, static_argnames=['gibbs_iters', 'max_clusters'])
def q_split(data, gibbs_iters, max_clusters, key, c0, alpha, g) -> Cluster:
    keys = jax.random.split(key, 3)
    c = (c0 + max_clusters * jax.random.bernoulli(keys[0], shape=c0.shape)).astype(int)
    # c = jnp.concatenate((jnp.zeros(100), jnp.ones(100) * 2))

    log_likelihood_mask = make_log_likelihood_mask(c0, max_clusters)

    partial_gibbs_step = partial(gibbs_step, 
        alpha=alpha, g=g, data=data, 
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

def gibbs_step(assignments, key, alpha, g, data, log_likelihood_mask, max_clusters):
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    f = gibbs_f(max_clusters, data, subkey1, g, assignments)
    pi = gibbs_pi(max_clusters, subkey2, alpha, assignments)

    assignments = gibbs_c(subkey3, pi, log_likelihood_mask, f, data)

    # now compute the logpdf for the assignments given the new distribution
    return assignments, Cluster(assignments, pi, f)

def gibbs_f(max_clusters, data, key, g, assignments):
    g_prime = posterior(g, data, assignments, 2*max_clusters)
    f = sample(key, g_prime)
    return f

def gibbs_c(key, pi, log_likelihood_mask, f, data):
    log_likelihoods = jax.vmap(jax.vmap(logpdf, in_axes=(0, None)), in_axes=(None, 0))(f, data)
    log_likelihoods = log_likelihoods + log_likelihood_mask
    log_score = log_likelihoods + jnp.log(pi)

    assignments = jax.random.categorical(key, log_score, axis=-1).astype(int)
    return assignments

def gibbs_pi(max_clusters, key, alpha, c):
    cluster_counts = jnp.sum(jax.nn.one_hot(c, num_classes=2 * max_clusters, dtype=jnp.int32), axis=0)
    pi = jax.random.dirichlet(key, alpha / 2 + cluster_counts)
    pi_pairs = pi.reshape((2, -1))
    pi_pairs = pi_pairs / jnp.sum(pi_pairs, axis=0)
    return pi_pairs.reshape(-1)