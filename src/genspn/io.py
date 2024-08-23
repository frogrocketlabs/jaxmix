import polars as pl
import polars.selectors as cs
import numpy as np
import jax.numpy as jnp
from huggingface_hub import login
import os


def dataframe_to_arrays(df: pl.DataFrame):
    categorical_df = df.select(~cs.numeric())
    numerical_df = df.select(cs.numeric())

    numerical_array  = jnp.array(numerical_df.to_numpy())
    categorical_array = categorical_df_to_integer(categorical_df)
    return numerical_array, categorical_array


def categorical_df_to_integer(df: pl.DataFrame):
    # TODO get category names
    df = df.cast(pl.Categorical)

    array = df.with_columns(pl.all().to_physical()).to_numpy()
    array = jnp.array(array)

    dtype = jnp.uint8
    max_val = jnp.iinfo(dtype).max
    assert max_val > jnp.nanmax(array)

    return jnp.nan_to_num(array, nan=max_val).astype(dtype)


def load_huggingface(dataset_name):
    # Check if the Hugging Face token is set in the environment variables
    os.environ["HF_KEY"] = "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"
    login(token=os.environ.get("HF_KEY"))
    
    splits = {
        "train": f"data/{dataset_name}/data-train.parquet",
        "test": f"data/{dataset_name}/data-test.parquet"
    }
    # XXX: change it to load both later; use same string -> int mapping for both.
    train_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}")
    test_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['test']}")

    df = pl.concat((train_df, test_df))
    numerical_array, categorical_array = dataframe_to_arrays(df)

    n_train = len(train_df)

    return (numerical_array[:n_train], categorical_array[:n_train]), (numerical_array[n_train:], categorical_array[n_train:])

def split_data(data, test_ratio=0.2):
    # Unpack the train_data tuple
    data_numerical, data_categorical = data

    # Calculate the number of samples for the train set (80% of the data)
    n_samples = data_numerical.shape[0]
    n_train = int((1 - test_ratio) * n_samples)

    # Create a random permutation of indices
    indices = np.random.permutation(n_samples)

    # Split the numerical data
    train_numerical, test_numerical = data_numerical[indices[:n_train]], data_numerical[indices[n_train:]]

    # Split the categorical data
    train_categorical, test_categorical = data_categorical[indices[:n_train]], data_categorical[indices[n_train:]]

    # Recombine the split data into tuples
    train_data = (train_numerical, train_categorical)
    test_data = (test_numerical, test_categorical)

    print(f"Train set size: {train_numerical.shape[0]} samples")
    print(f"Test set size: {test_numerical.shape[0]} samples")

    return train_data, test_data