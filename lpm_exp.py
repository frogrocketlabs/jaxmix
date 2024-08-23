# %%
%load_ext autoreload
%autoreload 2

# %%
from genspn.io import load_huggingface, split_data

train_data, valid_data = load_huggingface("AutoML/heart-disease")

# %%
train_data, test_data = split_data(train_data)

# %%
from genspn.distributions import make_trace
import jax.numpy as jnp
import jax
max_clusters = 50

alpha = 2
d = .1

key = jax.random.PRNGKey(1234)
key, subkey = jax.random.split(key)
trace = make_trace(subkey, alpha, d, train_data, max_clusters)

# %%
from genspn.smc import smc
gibbs_iters = 20

key, subkey = jax.random.split(key)
trace = smc(subkey, trace, test_data, max_clusters, train_data, gibbs_iters, max_clusters)

# %%
len(trace.cluster.pi)

# %%
from genspn.distributions import MixtureModel, logpdf
cluster = trace.cluster[8]

mixture_model = MixtureModel(
    pi=cluster.pi/jnp.sum(cluster.pi), 
    f=cluster.f[:max_clusters])
logprobs = jax.vmap(logpdf, in_axes=(None, 0))(mixture_model, valid_data)
