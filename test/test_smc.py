import jax
import jax.numpy as jnp

from genspn.smc import q_split, split_cluster, step
from genspn.distributions import NormalInverseGamma, Dirichlet, MixedConjugate, posterior, sample, logpdf, Categorical, Cluster, Trace, GEM

def test_split_cluster():
    max_clusters = jnp.array(3)
    n = 4
    k = jnp.array(1)
    K = jnp.array(3)

    c0 = jnp.array([0, 0, 1, 1])
    pi0 = jnp.array([.5, .5, 0, 0, 0, 0])

    c1 = jnp.array([0, 3, 1, 4])
    pi1 = jnp.array([.2, .3, 0, .8, .7, 0])
    f0 = Categorical(
        jnp.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.5, 0.5],
        ])
    )
    f1 = Categorical(
        jnp.array([
            [0.2, 0.8],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.4, 0.6],
        ])
    )

    cluster0 = Cluster(c=c0, pi=pi0, f=f0)
    split_clusters = Cluster(c=c1, pi=pi1, f=f1)

    new_cluster = split_cluster(cluster0, split_clusters, k, K, max_clusters)

    assert jnp.array_equal(new_cluster.c, jnp.array([0, 0, 1, 2]))
    assert jnp.array_equal(new_cluster.pi, jnp.array([.5, .15, .35, 0, 0, 0]))
    assert jnp.array_equal(new_cluster.f.logprobs, jnp.array([
        [.1, .9],
        [.1, .9],
        [.4, .6],
        [0.4, 0.6],
        [0.5, 0.5],
        [0.5, 0.5],
        ]))

def test_smc():
    iters = 20
    alpha = 1
    d = .1
    key = jax.random.PRNGKey(1234)
    keys = jax.random.split(key, 8)
    max_clusters = 3
    n_data0 = jax.random.normal(keys[0], (100, 2)) * .1
    n_data1 = 2 + .1 * jax.random.normal(keys[1], (100, 2))
    n_data2 = -2 + .1 * jax.random.normal(keys[1], (100, 2))

    keys_c0 = jax.random.split(keys[2], 100)
    keys_c1 = jax.random.split(keys[3], 100)
    keys_c2 = jax.random.split(keys[4], 100)
    c_data0 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c0, jnp.log(jnp.array([0.8, 0.2])))
    c_data1 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c1, jnp.log(jnp.array([0.1, 0.9])))
    c_data2 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c2, jnp.log(jnp.array([0.4, 0.6])))

    data = jnp.concatenate([n_data0, n_data1, n_data2], axis=0), jnp.concatenate([c_data0, c_data1, c_data2], axis=0)[:, None]

    nig = NormalInverseGamma(m=jnp.zeros(2), l=jnp.ones(2), a=jnp.ones(2), b=jnp.ones(2))
    dirichlet = Dirichlet(alpha=jnp.ones((1, 2)))
    g = MixedConjugate(nig=nig, dirichlet=dirichlet)

    c = jnp.zeros(300, dtype=int)
    g_prime = posterior(g, data, c, 2 * max_clusters)
    f = sample(keys[5], g_prime)
    pi = jnp.array([.9, 0, 0, 0, 0, 0,])
    cluster = Cluster(c=c, f=f, pi=pi)
    gem = GEM(alpha=alpha, d=d)

    trace = Trace(gem=gem, g=g, cluster=cluster)

    for i in range(2):
        # sample random assignments to begin with
        # note: we should flush the cs so they're 1:N
        new_cluster = step(data, trace=trace, gibbs_iters=iters, max_clusters=3, key=keys[6 + i], K=i+2)
        trace = Trace(
            gem=gem,
            g=g,
            cluster=new_cluster
        )

    assert jnp.all(trace.cluster.c[:100] == trace.cluster.c[0])
    assert jnp.all(trace.cluster.c[100:200] == trace.cluster.c[100])
    assert jnp.all(trace.cluster.c[200:] == trace.cluster.c[200])
    assert trace.cluster.c[0] != trace.cluster.c[100]
    assert trace.cluster.c[200] != trace.cluster.c[100]
    assert trace.cluster.c[200] != trace.cluster.c[0]
