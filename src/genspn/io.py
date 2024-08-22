import polars as pl
import polars.selectors as cs
import numpy as np
from huggingface_hub import login
import os


def dataframe_to_arrays(df: pl.DataFrame):
    categorical_df = df.select(~cs.numeric())
    numerical_df = df.select(cs.numeric())

    categorical_df = categorical_df_to_integer(categorical_df)
    return numerical_df.to_numpy(), categorical_df.to_numpy()

def categorical_df_to_integer(df: pl.DataFrame):
    # TODO get category names
    df = df.cast(pl.Categorical)
    return df.with_columns(pl.all().to_physical())

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