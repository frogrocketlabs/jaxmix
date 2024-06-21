from genspn.distributions import logpdf, GEM, Normal
import jax.numpy as jnp
import jax


def test_score_stick_breaking():
    gem = GEM(1, 0)
    pis0 = jnp.array([1/4, 1/4, 1/4])
    pis1 = jnp.array([1/2, 1/4, 1/8])

    logp0 = logpdf(gem, pis0)
    logp1 = logpdf(gem, pis0)

    pass


def test_score_data():
    x = jnp.zeros((5, 6))
    c = jnp.array([0, 1, 0, 1, 0], dtype=int)

    dist  = Normal(
        mu=jnp.vstack((
            jnp.zeros(6), 
            jnp.ones(6))), 
        std=jnp.vstack((
            jnp.ones(6),
            jnp.ones(6),
        )))

    x_logpdfs = jax.vmap(logpdf, in_axes=(None, 0, 0))(dist, x, c)
    assert x_logpdfs[0] == x_logpdfs[2]
    assert x_logpdfs[2] == x_logpdfs[5]
    assert x_logpdfs[1] == x_logpdfs[3]
    assert x_logpdfs[0] != x_logpdfs[1]