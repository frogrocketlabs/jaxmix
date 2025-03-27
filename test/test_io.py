import jax.numpy as jnp
import os

from jaxmix.io import serialize
from jaxmix.io import deserialize

MIXTURE_PARAMETERS = {
    "mu": jnp.array([[[0.0, 1.0]]]),
    "sigma": jnp.array([[[1.0, 2.0]]]),
    "logprobs": jnp.log(jnp.array([[[0.2, 0.8]]])),
    "cluster_weights": jnp.array([1.0]),
}


def _remove_path(path):
    os.remove(path)
    assert not os.path.exists(path), "Sanity check."


PATH = "temp-test-file.model"


def test_serialize():
    serialize(MIXTURE_PARAMETERS, PATH)
    assert os.path.exists(PATH)
    _remove_path(PATH)


def test_deserialize():
    serialize(MIXTURE_PARAMETERS, PATH)
    assert os.path.exists(PATH), "Sanity check."
    mixture_parameters_loaded = deserialize(PATH)
    for k, arr in MIXTURE_PARAMETERS.items():
        assert jnp.array_equal(arr, mixture_parameters_loaded[k])
    _remove_path(PATH)
