from genspn.distributions import Normal, Categorical, Dirichlet, Mixed, logpdf, posterior
import jax.numpy as jnp


def test_posterior_dirichlet():
    n_dim = 2
    k = 3
    N = 4
    alphas = jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, -jnp.inf]])
    dirichlet = Dirichlet(alphas)

    # x = jnp.array([[0, 1], [1, 0], [2, 0], [0, 0]])
    counts = jnp.array([
        [2, 1, 1],
        [3, 1, 0]
    ])
    # 0, 0, 1, 2 for the first dim
    # 0, 0, 0, 1 for the second dim
    dirichlet_posterior = posterior(dirichlet, counts)
    assert jnp.all(dirichlet_posterior.alpha[0] == jnp.array([1 + 2, 2 + 1, 3 + 1]))
    assert jnp.all(dirichlet_posterior.alpha[1] == jnp.array([3 + 3, 4 + 1, -jnp.inf]))


def test_posterior_nic2():
    pass

def test_normal():
    mu = jnp.array([0.0, 1.0])
    std = jnp.array([1.0, 2.0])

    params = Normal(mu=mu, std=std)

    logp = logpdf(params, jnp.array([0.0, 0.0]))

    logp0 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(1.0) - 0.5 * ((0.0 - 0.0) / 1.0) ** 2
    logp1 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(2.0) - 0.5 * ((0.0 - 1.0) / 2.0) ** 2
    assert logp == logp0 + logp1

def test_categorical():
    probs = jnp.array([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]])
    logprobs = jnp.log(probs)

    params = Categorical(logprobs=logprobs)

    logp = logpdf(params, jnp.array([0, 1]))

    logp0 = jnp.log(0.4)
    logp1 = jnp.log(0.3)

    assert logp == logp0 + logp1

def test_mixed():
    mu = jnp.array([0.0, 1.0])
    std = jnp.array([1.0, 2.0])

    normal_params = Normal(mu=mu, std=std)

    probs = jnp.array([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]])
    logprobs = jnp.log(probs)

    categorical_params = Categorical(logprobs=logprobs)

    mixed_params = Mixed(normal=normal_params, categorical=categorical_params)

    logp = logpdf(mixed_params, (jnp.array([0.0, 0.0]), jnp.array([0, 1])))

    logp_n0 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(1.0) - 0.5 * ((0.0 - 0.0) / 1.0) ** 2
    logp_n1 = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(2.0) - 0.5 * ((0.0 - 1.0) / 2.0) ** 2
    
    logp_c0 = jnp.log(0.4)
    logp_c1 = jnp.log(0.3) 

    assert logp == logp_n0 + logp_n1 + logp_c0 + logp_c1