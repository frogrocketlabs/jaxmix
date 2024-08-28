import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array, Float, Integer
from plum import dispatch
from typing import Optional
from numbers import Real

ZERO = 1e-20

class NormalInverseGamma(eqx.Module):
    m: Float[Array, "*batch n_dim"]
    l: Float[Array, "*batch n_dim"]
    a: Float[Array, "*batch n_dim"]
    b: Float[Array, "*batch n_dim"]

class Dirichlet(eqx.Module):
    alpha: Float[Array, "*batch n_dim k"]

    def __getitem__(self, key):
        return Dirichlet(alpha=self.alpha[key])

class Normal(eqx.Module):
    mu: Float[Array, "*batch n_dim"]
    std: Float[Array, "*batch n_dim"]

    def __getitem__(self, key):
        return Normal(mu=self.mu[key], std=self.std[key])

class Categorical(eqx.Module):
    # assumed normalized, padded
    logprobs: Float[Array, "*batch n_dim k"]
    def __getitem__(self, key):
        return Categorical(logprobs=self.logprobs[key])

class Mixed(eqx.Module):
    normal: Normal
    categorical: Categorical

    def __getitem__(self, key):
        return Mixed(normal=self.normal[key], categorical=self.categorical[key])

class MixedConjugate(eqx.Module):
    nig: NormalInverseGamma
    dirichlet: Dirichlet

class GEM(eqx.Module):
    alpha: Float[Array, "*batch"]
    d: Float[Array, "*batch"]

class Cluster(eqx.Module):
    c: Float[Array, "*batch n"]
    pi: Float[Array, "*batch k"]
    f: Float[Array, "*batch k"]

    def __getitem__(self, key):
        return Cluster(self.c[key], self.pi[key], self.f[key])

class Trace(eqx.Module):
    gem: GEM
    g: NormalInverseGamma | Dirichlet | MixedConjugate
    cluster: Cluster

type F = Categorical | Normal | Mixed

type Datapoint = Float[Array, "n_c"] | Integer[Array, "n_d"] | tuple[Float[Array, "n_c"], Integer[Array, "n_d"]]

class MixtureModel(eqx.Module):
    # mask: Datapoint
    pi: Float[Array, "*batch k"]
    f: F

@dispatch
def marginalize(dist: Normal | Categorical, idxs: tuple[int, ...]) -> Normal | Categorical:
    return dist[idxs]

@dispatch
def marginalize(dist: Mixed, idxs: tuple[tuple[int, ...]]) -> Mixed:
    return Mixed(normal=dist.normal[idxs[0]], categorical=dist.categorical[idxs[1]])

@dispatch
def sample(key: Array, dist: Dirichlet) -> Categorical:
    probs = jax.random.dirichlet(key, dist.alpha)
    probs = jnp.where(probs == 0, ZERO, probs)

    return Categorical(jnp.log(probs))

@dispatch 
def sample(key: Array, dist: NormalInverseGamma) -> Normal:
    """ See Kevin Murphy's Conjugate Bayesian analysis of the Gaussian distribution:
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf """
    keys = jax.random.split(key)

    log_lambda = jax.random.loggamma(key, dist.a) - jnp.log(dist.b)
    log_sigma = -jnp.log(dist.l) - log_lambda
    std = jnp.exp(log_sigma/ 2)
    mu = dist.m + jax.random.normal(keys[1], shape=dist.m.shape) * std

    return Normal(mu=mu, std=jnp.exp(-log_lambda/2))

@dispatch
def sample(key: Array, dist: MixedConjugate) -> Mixed:
    keys = jax.random.split(key)
    normal = sample(keys[0], dist.nig)
    categorical = sample(keys[1], dist.dirichlet)

    return Mixed(normal=normal, categorical=categorical)

@dispatch
def sample(key: Array, dist: Normal) -> Float[Array, "n_c"]:
    return dist.mu + dist.std * jax.random.normal(key, shape=dist.mu.shape)

@dispatch
def sample(key: Array, dist: Categorical) -> Integer[Array, "n_D"]:
    return jax.random.categorical(key, dist.logprobs)

@dispatch
def sample(key: Array, dist: Mixed) -> tuple[Float[Array, "n_c"], Integer[Array, "n_d"]]:
    keys = jax.random.split(key)
    normal = sample(keys[0], dist.normal)
    categorical = sample(keys[1], dist.categorical)

    return normal, categorical

@dispatch 
def sample(key: Array, dist: MixtureModel):
    keys = jax.random.split(key)
    cluster = jax.random.categorical(keys[0], dist.pi)
    return sample(key, dist.f[cluster])

@dispatch
def posterior(dist: MixedConjugate, x: tuple[Float[Array, "batch n_normal_dim"], Integer[Array, "batch n_categorical_dim"]]) -> MixedConjugate:
    nig = posterior(dist.nig, x[0])
    dirichlet = posterior(dist.dirichlet, x[1])

    return MixedConjugate(nig=nig, dirichlet=dirichlet)

@dispatch
def posterior(dist: MixedConjugate, x: tuple[Float[Array, "batch n_normal_dim"], Integer[Array, "batch n_categorical_dim"]], c: Integer[Array, "batch"], max_clusters:Optional[int]=None) -> MixedConjugate:
    nig = posterior(dist.nig, x[0], c, max_clusters)
    dirichlet = posterior(dist.dirichlet, x[1], c, max_clusters)

    return MixedConjugate(nig=nig, dirichlet=dirichlet)

@dispatch
def posterior(dist: NormalInverseGamma, x: Float[Array, "batch n_dim"], c: Integer[Array, "batch"], max_clusters:Optional[int]=None) -> NormalInverseGamma:
    N = jax.ops.segment_sum(jnp.ones(x.shape[0], dtype=jnp.int32), c, num_segments=max_clusters)
    masked_x = jnp.nan_to_num(x, 0.)
    sum_x = jax.ops.segment_sum(masked_x, c, num_segments=max_clusters)
    sum_x_sq = jax.ops.segment_sum(masked_x ** 2, c, num_segments=max_clusters)

    return jax.vmap(posterior, in_axes=(None, 0, 0, 0))(dist, N, sum_x, sum_x_sq)

@dispatch
def posterior(dist: NormalInverseGamma, x: Float[Array, "batch n_dim"]) -> NormalInverseGamma:
    N = x.shape[0]
    sum_x = jnp.nansum(x, axis=0)
    sum_x_sq = jnp.nansum(x ** 2, axis=0)

    return posterior(dist, N, sum_x, sum_x_sq)

@dispatch
def posterior(dist: NormalInverseGamma, N: Integer[Array, ""], sum_x: Float[Array, "n_dim"], sum_x_sq: Float[Array, "n_dim"]) -> NormalInverseGamma:
    l = dist.l + N
    m = (dist.l * dist.m + sum_x) / l
    a = dist.a + N / 2
    b = dist.b + 0.5 * (sum_x_sq + dist.l * dist.m ** 2 - l * m ** 2)

    return NormalInverseGamma(m=m, l=l, a=a, b=b)

@dispatch
def posterior(dist: Dirichlet, x: Integer[Array, "batch n_dim"], c: Integer[Array, "batch"], max_clusters:Optional[int]=None) -> Dirichlet:
    one_hot_x = jax.nn.one_hot(x, num_classes=dist.alpha.shape[-1], dtype=jnp.int32)
    counts = jax.ops.segment_sum(one_hot_x, c, num_segments=max_clusters)
    return jax.vmap(posterior, in_axes=(None, 0))(dist, counts)

@dispatch
def posterior(dist: Dirichlet, counts: Integer[Array, "n_dim k"]) -> Dirichlet:
    return Dirichlet(alpha=dist.alpha + counts)

@dispatch
def logpdf(dist: Normal, x: Float[Array, "n_dim"]) -> Float[Array, ""]:
    logprob = jnp.nansum(-0.5 * jnp.log(2 * jnp.pi) - jnp.log(dist.std) - 0.5 * ((x - dist.mu) / dist.std) ** 2)

    return logprob

@dispatch
def logpdf(dist: Categorical, x: Integer[Array, "n_dim"]) -> Float[Array, ""]:
    return jnp.nansum(dist.logprobs.at[jnp.arange(x.shape[-1]), x].get(mode="fill", fill_value=jnp.nan))

@dispatch
def logpdf(dist: Mixed, x: tuple[Float[Array, "n_normal_dim"], Integer[Array, "n_categorical_dim"]]) -> Float[Array, ""]:
    return logpdf(dist.normal, x[0]) + logpdf(dist.categorical, x[1])

@dispatch
def logpdf(dist: GEM, pi: Float[Array, "n"], K: Integer[Array, ""]) -> Float[Array, ""]:
    betas = jax.vmap(lambda i: 1 - pi[i] / pi[i-1])(jnp.arange(len(pi)))
    betas = betas.at[0].set(pi[0])
    logprobs = jax.vmap(jax.scipy.stats.beta.logpdf, in_axes=(0, None, 0))(betas, 1-dist.d, dist.alpha + (1 + jnp.arange(len(pi))) * dist.d)
    idx = jnp.arange(logprobs.shape[0])
    logprobs = jnp.where(idx < K, logprobs, 0) 
    return jnp.sum(logprobs)

@dispatch
def logpdf(dist: F, x: Datapoint, c: Integer[Array, ""]) -> Float[Array, ""]:
    dist = dist[c]
    return logpdf(dist, x)

@dispatch
def logpdf(dist: MixtureModel, x: Datapoint) -> Float[Array, ""]:
    logprob = jax.vmap(logpdf, in_axes=(0, None))(dist.f, x)
    logprob = logprob + jnp.log(dist.pi)
    return jax.scipy.special.logsumexp(logprob)

@dispatch
def logpdf(dist: MixedConjugate, x: Mixed)-> Float[Array, ""]:
    return logpdf(dist.nig, x.normal) + logpdf(dist.dirichlet, x.categorical)

@dispatch
def logpdf(dist: NormalInverseGamma, x: Normal)-> Float[Array, ""]:
    std_logpdf = jax.scipy.stats.gamma.logpdf(x.std ** -2, dist.a, scale=1/dist.b)
    mu_logpdf = jax.scipy.stats.norm.logpdf(x.mu, loc=dist.m, scale=x.std / jnp.sqrt(dist.l))
    return jnp.sum(mu_logpdf + std_logpdf)

@dispatch
def logpdf(dist: Dirichlet, x: Categorical)-> Float[Array, ""]:
    logprobs = jax.vmap(jax.scipy.stats.dirichlet.logpdf)(jnp.exp(x.logprobs), dist.alpha)
    return jnp.sum(logprobs)

def make_trace(
        key: jax.Array, alpha: Real, d: Real, 
        data: tuple[Float[Array, "n n_c"], Integer[Array, "n n_d"]] | Float[Array, "n n_c"] | Integer[Array, "n n_d"], 
        max_clusters: int):

    g = make_g(data)

    n = len(data[0]) if isinstance(data, tuple) else len(data)
    c = jnp.zeros(n, dtype=int)

    g_prime = posterior(g, data, c, 2 * max_clusters)

    f = sample(key, g_prime)
    pi = jnp.zeros(max_clusters)
    pi = pi.at[0].set(.9)
    cluster = Cluster(c=c, f=f, pi=pi)
    gem = GEM(alpha=alpha, d=d)

    return Trace(gem=gem, g=g, cluster=cluster)

@dispatch
def make_g(data: tuple[Float[Array, "n n_c"], Integer[Array, "n n_d"]]):
    nig = make_g(data[0])
    dirichlet = make_g(data[1])

    return MixedConjugate(nig=nig, dirichlet=dirichlet)


@dispatch
def make_g(data: Float[Array, "n n_c"]):
    n_continuous = data.shape[1]

    return NormalInverseGamma(
        m=jnp.zeros(n_continuous), l=jnp.ones(n_continuous), 
        a=jnp.ones(n_continuous), b=jnp.ones(n_continuous))

@dispatch
def make_g(data: Integer[Array, "n n_d"]):
    n_discrete = data.shape[1]
    n_categories = jnp.nanmax(data, axis=0) + 1
    max_n_categories = jnp.max(n_categories).astype(int)

    cat_alpha = jnp.ones((n_discrete, max_n_categories))
    mask = jnp.tile(jnp.arange(max_n_categories), (n_discrete, 1)) <= n_categories[:, None]
    cat_alpha = jnp.where(mask, cat_alpha, ZERO)

    return Dirichlet(alpha=cat_alpha)