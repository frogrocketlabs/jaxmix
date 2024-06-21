import jax
import jax.numpy as jnp

from genspn.smc import step
from genspn.distributions import NormalInverseGamma, Dirichlet, MixedConjugate, posterior, sample, logpdf


def test_smc():
    iters = 50
    alpha = 1
    key = jax.random.PRNGKey(1234)
    keys = jax.random.split(key, 7)
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
    h = MixedConjugate(nig=nig, dirichlet=dirichlet)
    theta = sample(keys[6], h)

    c = jnp.zeros(300)

    for i in range(2):
        # sample random assignments to begin with
        # note: we should flush the cs so they're 1:N
        theta, c = step(alpha, h, data, key, c)

    assert jnp.all(c[:100] == c[0])
    assert jnp.all(c[100:200] == c[100])
    assert jnp.all(c[200:] == c[200])
    assert c[0] != c[100]
    assert c[200] != c[100]
    assert c[200] != c[0]
