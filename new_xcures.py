# %%
from genspn.io import csv_to_array
import jax.numpy as jnp
import jax
from genspn.distributions import ZERO, Dirichlet, MixedConjugate, posterior, sample, logpdf, Categorical, Cluster, Trace, GEM, MixtureModel
from genspn.smc import smc

# %%
from genspn.io import csv_to_array
import jax.numpy as jnp

categorical_vars = [
    "race",
    "ethnicity",
    "sex_at_birth",
    "cancer_type",
    "deceased",
    "performance_exam_score",
    "dexamethasone",
    "prednisone",
    "celecoxib",
    "anastrozole",
    "methylprednisolone",
]
data = csv_to_array("data/xcures.csv", categorical_vars)

data[0][:, 1] = jnp.sqrt(jnp.where(data[0][:, 1] <= 0, jnp.nan, data[0][:, 1]))
data[0] = (data[0] - jnp.nanmean(data[0], axis=0)) 
data[0] = data[0] / jnp.nanstd(data[0], axis=0)

n_categories = jnp.nanmax(data[1], axis=0) + 1
dtype = jnp.uint8
max_val = jnp.iinfo(dtype).max
assert max_val > jnp.nanmax(data[1])
data[1] = jnp.nan_to_num(data[1], nan=max_val).astype(dtype)

data = tuple(data)

# %%
n = data[0].shape[0]
n_continuous = data[0].shape[1]
n_discrete = data[1].shape[1]
n_continuous
n_discrete

# %%
max_n_categories = jnp.max(n_categories).astype(int)

# %%
import jax
from genspn.distributions import Dirichlet, NormalInverseGamma, MixedConjugate, posterior, sample, Cluster, Trace, GEM

n_test = 1000
key = jax.random.PRNGKey(1234)
subkey, key = jax.random.split(key)

test_idxs = jax.random.choice(
    subkey, jnp.arange(n), shape=(n_test,), replace=False
)
data_test = (data[0][test_idxs], data[1][test_idxs])
data = (jnp.delete(data[0], test_idxs, axis=0), jnp.delete(data[1], test_idxs, axis=0))

gibbs_iters = 20
smc_steps = 25
max_clusters = 50
alpha = 2
d = .1

nig = NormalInverseGamma(
    m=jnp.zeros(n_continuous), l=jnp.ones(n_continuous), 
    a=jnp.ones(n_continuous), b=jnp.ones(n_continuous))

cat_alpha = jnp.ones((n_discrete, max_n_categories))
mask = jnp.tile(jnp.arange(max_n_categories), (n_discrete, 1)) <= n_categories[:, None]
cat_alpha = jnp.where(mask, cat_alpha, ZERO)

dirichlet = Dirichlet(alpha=cat_alpha)
g = MixedConjugate(nig=nig, dirichlet=dirichlet)

c = jnp.zeros(len(data[0]), dtype=int)

# %%
g_prime = posterior(g, data, c, 2 * max_clusters)

# %%
f = sample(key, g_prime)
pi = jnp.zeros(max_clusters)
pi = pi.at[0].set(.9)
cluster = Cluster(c=c, f=f, pi=pi)
gem = GEM(alpha=alpha, d=d)

trace = Trace(gem=gem, g=g, cluster=cluster)

# %%
from genspn.smc import smc
trace = smc(key, trace, data_test, n_steps=smc_steps, data=data, gibbs_iters=gibbs_iters, max_clusters=max_clusters)

# %%

idx = 17
mix = MixtureModel(pi=trace.cluster.pi[idx]/jnp.sum(trace.cluster.pi[idx]), f=trace.cluster.f[idx][:max_clusters])
# %%
data = csv_to_array("data/xcures.csv", categorical_vars)

data[0][:, 1] = jnp.sqrt(jnp.where(data[0][:, 1] <= 0, jnp.nan, data[0][:, 1]))
mu = jnp.nanmean(data[0], axis=0)
std = jnp.nanstd(data[0] - mu, axis=0)

# %%
import polars as pl
df =  pl.read_csv("data/xcures.csv") 

# %%
df

# %%
x_c = jnp.nan * jnp.ones(2)
x_d = 255 * jnp.ones(11, dtype=int)
logpdf(mix, (x_c, x_d))
# %%
data
# %%
logpdf(mix.f, (x_c, x_d))
# %%
logpdf(mix.f.normal, x_c)
# %%
logpdf(mix.f.categorical, x_d)
# %%
mix.f.categorical
# %%
jnp.nansum(mix.f.categorical.logprobs.at[, 255].get(mode="fill", fill_value=jnp.nan))
# %%
df.describe()
# %%
survival_time = 1000 * jnp.arange(1, 10)
# %%
transformed_survival_time = (jnp.sqrt(survival_time) - mu[1]) / std[1]

# %%
def survival_time_greater_than(x):
    transformed_x = (jnp.sqrt(x) - mu[1]) / std[1]
    cdfs = 1 - jax.vmap(jax.scipy.stats.norm.cdf, in_axes=(None, 0, 0))(
        transformed_x, mix.f.normal.mu[:, 1], mix.f.normal.std[:, 1]
    )

    return {"days":  x, "survival probability": jnp.sum(mix.pi * cdfs)}

# %%
import numpy as np
survival_time_probs = jax.vmap(survival_time_greater_than)(survival_time)
survival_time_probs = {k: np.array(v) for k, v in survival_time_probs.items()}

# %%
results_df = pl.from_dict(survival_time_probs)

# %%
categorical_df = df.select(categorical_vars)
categorical_df = categorical_df.cast(pl.Categorical)

cat_dict = {cat: categorical_df[cat].cat.get_categories()
for cat in treatments}
cat_dict

# %%
import altair as alt

# %%
alt.Chart(results_df).mark_line().encode(
    x="days",
    y="survival probability"
)
# %%
treatments = ["dexamethasone",
    "prednisone",
    "celecoxib",
    "anastrozole",
    "methylprednisolone",
]


def treatment_conditional(treatment_idx):
    var_idx = 5 + treatment_idx
    x_c = jnp.nan * jnp.ones(2)
    x_d = 255 * jnp.ones(11, dtype=int)
    yes_idx = jnp.where(treatment_idx <= 1, 0., 1.)
    x_d = x_d.at[var_idx].set(yes_idx)

    pi_logpdf = jax.vmap(logpdf, in_axes=(0, None))(mix.f, (x_c, x_d))    
    new_pi = mix.pi * jnp.exp(pi_logpdf)
    new_pi = new_pi / jnp.sum(new_pi)
    return MixtureModel(pi=new_pi, f=mix.f)

conditional_models = jax.vmap(treatment_conditional)(jnp.arange(len(treatments)))

# %%
conditional_models

# %%

def survival_time_greater_than(x, model_idx):
    model = MixtureModel(
        pi = conditional_models.pi[model_idx],
        f = conditional_models.f[model_idx]
    )
    transformed_x = (jnp.sqrt(x) - mu[1]) / std[1]
    cdfs = 1 - jax.vmap(jax.scipy.stats.norm.cdf, in_axes=(None, 0, 0))(
        transformed_x, model.f.normal.mu[:, 1], model.f.normal.std[:, 1]
    )

    return {"days":  x, "survival probability": jnp.sum(model.pi * cdfs), "treatment": model_idx}

# %%
results_dict = jax.vmap(
    jax.vmap(survival_time_greater_than, in_axes=(0, None)),
    in_axes=(None, 0))(survival_time, jnp.arange(5))
# %%
results_dict = {k: np.array(v).ravel() for k, v in results_dict.items()}

# %%
results_df = pl.from_dict(results_dict)

# %%
alt.Chart(results_df).mark_line().encode(
    x="days",
    y="survival probability",
    color="treatment:N"
)
# %%
results_df = results_df.with_columns(
    treatment=pl.col("treatment").replace([0, 1, 2, 3, 4], treatments)
)

# %%
transformed_age = (70 - mu[0]) / std[0]
categorical_df["race"].cat.get_categories()
# %%
black_idx = 2
race_idx = 0
x_c = jnp.nan * jnp.ones(2)
x_d = 255 * jnp.ones(11, dtype=int)
x_d = x_d.at[race_idx].set(black_idx)
x_c = x_c.at[0].set(transformed_age)

pi_logpdf = jax.vmap(logpdf, in_axes=(0, None))(mix.f, (x_c, x_d))    
new_pi = mix.pi * jnp.exp(pi_logpdf)
new_pi = new_pi / jnp.sum(new_pi)
black70 = MixtureModel(pi=new_pi, f=mix.f)
