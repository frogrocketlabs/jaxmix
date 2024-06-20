import jax.numpy as jnp
import jax
from genspn.smc import step
from genspn.distributions import Normal, Categorical, Dirichlet, NormalInverseGamma, Mixed, MixedConjugate, posterior, sample, logpdf
from functools import partial


def test_mixture_model():
    iters = 20
    alpha = 1
    key = jax.random.PRNGKey(1234)
    keys = jax.random.split(key, 6)
    n_data0 = jax.random.normal(keys[0], (100, 2)) * .1
    n_data1 = 1 + .1 * jax.random.normal(keys[1], (100, 2))

    keys_c0 = jax.random.split(keys[2], 100)
    keys_c1 = jax.random.split(keys[3], 100)
    c_data0 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c0, jnp.log(jnp.array([0.4, 0.6])))
    c_data1 = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys_c1, jnp.log(jnp.array([0.4, 0.6])))

    data = jnp.concatenate([n_data0, n_data1], axis=0), jnp.concatenate([c_data0, c_data1], axis=0)[:, None]
    c = jnp.zeros(200)
    nig = NormalInverseGamma(m=jnp.zeros(2), l=jnp.ones(2), a=jnp.ones(2), b=jnp.ones(2))
    dirichlet = Dirichlet(alpha=jnp.ones((1, 2)))
    h = MixedConjugate(nig=nig, dirichlet=dirichlet)

    # step_jit = jax.jit(partial(step, gibbs_iters=iters))
    # c, pi, theta = step_jit(alpha, h, data, keys[5], c, gibbs_iters=20)
    c, pi, theta = step(alpha, h, data, keys[5], c, gibbs_iters=20)

    assert jnp.all(c[-1, :100] == c[-1, 0])
    assert jnp.all(c[-1, 100:] == c[-1, 100])
    assert c[-1, 0] != c[-1, 100]