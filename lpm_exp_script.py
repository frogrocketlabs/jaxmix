import jax
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
from genspn.io import load_huggingface, split_data
from genspn.distributions import make_trace
import jax.numpy as jnp
from genspn.smc import smc
from genspn.distributions import MixtureModel, logpdf
from functools import partial
from pathlib import Path
from huggingface_hub import login, HfApi
from tqdm import tqdm
import os
import numpy as np
import polars as pl


def run_experiment(max_clusters, gibbs_iters, alpha, d, key, train_data, test_data, valid_data):
    key, subkey = jax.random.split(key)
    trace = make_trace(subkey, alpha, d, train_data, max_clusters)

    key, subkey = jax.random.split(key)
    trace, sum_logprobs = smc(subkey, trace, test_data, max_clusters, train_data, gibbs_iters, max_clusters)

    idx = jnp.argmax(sum_logprobs)
    cluster = trace.cluster[idx]

    mixture_model = MixtureModel(
        pi=cluster.pi/jnp.sum(cluster.pi), 
        f=cluster.f[:max_clusters])
    return jax.vmap(logpdf, in_axes=(None, 0))(mixture_model, valid_data)

def run_experiment_wrapper(key, dataset_name, n_replicates, max_clusters, gibbs_iters, alpha, d):
    keys = jax.random.split(key, n_replicates)
    train_data, valid_data = load_huggingface(dataset_name)
    train_data, test_data = split_data(train_data, .1)

    partial_run_exp = partial(run_experiment, max_clusters, gibbs_iters, alpha, d)

    logprobs = jax.vmap(partial_run_exp, in_axes=(0, None, None, None))(
        keys, train_data, test_data, valid_data)

    data_id = jnp.arange(logprobs.shape[1])
    data_id = jnp.tile(data_id, (n_replicates, 1))

    replicate = jnp.arange(logprobs.shape[0])
    replicate = jnp.tile(replicate, (logprobs.shape[1], 1)).T

    df = pl.DataFrame({
        "data_id": np.array(data_id.flatten()),
        "replicate": np.array(replicate.flatten()),
        "logprob": np.array(logprobs.flatten())
    })

    path = Path("results", dataset_name, "held-out-likelihood.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

os.environ["HF_KEY"] = "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"
login(token=os.environ.get("HF_KEY"))

api = HfApi()
dataset = api.dataset_info("Large-Population-Model/model-building-evaluation")
config = dataset.card_data['configs']

# temporary: filter out LPM datasets
dataset_names = [c['config_name'].replace("-", "/", 1) 
    for c in config if "LPM" not in c['config_name']]
n_replicates = 10
key = jax.random.PRNGKey(1234)
max_clusters = 50
alpha = 2
d = .1
gibbs_iters = 20

for dataset_name in tqdm(dataset_names):
    run_experiment_wrapper(key, dataset_name, n_replicates, max_clusters, gibbs_iters, alpha, d)