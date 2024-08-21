# %%
import polars as pl
import pandas as pd
from huggingface_hub import login
import os

def load_dataset(dataset_name):
    # Check if the Hugging Face token is set in the environment variables
    os.environ["HF_KEY"] = "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"
    login(token=os.environ.get("HF_KEY"))
    
    splits = {
        "train": f"data/{dataset_name}/data-train.parquet",
        "test": f"data/{dataset_name}/data-test.parquet"
    }
    # XXX: change it to load both later; use same string -> int mapping for both.
    df = pd.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}")

    numerical_df = pl.DataFrame(df.select_dtypes(include=['number']))
    categorical_df = pl.DataFrame(df.select_dtypes(exclude=['number']))


    categorical_df = categorical_df.cast(pl.Categorical)
    numbered_categorical_df = categorical_df.with_columns(pl.all().to_physical())
    return [numerical_df.to_numpy(), numbered_categorical_df.to_numpy()]

# Example usage:
numerical_array, categorical_array = load_dataset("AutoML/heart-disease")
# %%
# Split the DataFrame into numerical and categorical dataframes


# %%
df.head()
# %%
