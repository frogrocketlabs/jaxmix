import polars as pl
import numpy as np

# convert csv to numpy array
def csv_to_array(file, categorical_vars):
    df = pl.read_csv(file)
    categorical_df = df.select(categorical_vars)
    numerical_df = df.select(pl.exclude(categorical_vars))

    categorical_df = categorical_df.cast(pl.Categorical)
    numbered_categorical_df = categorical_df.with_columns(pl.all().to_physical())

    return [numerical_df.to_numpy(), numbered_categorical_df.to_numpy()]