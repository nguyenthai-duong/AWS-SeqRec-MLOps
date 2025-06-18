import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm.auto import tqdm


def chunk_transform(df, pipeline, chunk_size=1000):
    transformed_chunks = []

    progress_bar = tqdm(range(0, df.shape[0], chunk_size), desc="Transforming chunks")

    # Iterate through the DataFrame in chunks
    for start in progress_bar:
        end = min(start + chunk_size, df.shape[0])
        chunk_df = df.iloc[start:end]

        # Apply the pipeline transformation to the chunk
        transformed_chunk = pipeline.transform(chunk_df)

        # Check if the transformed output is sparse, and convert to dense
        if issparse(transformed_chunk):
            transformed_chunk = transformed_chunk.toarray()

        # Collect the transformed chunk
        transformed_chunks.append(transformed_chunk)

    # Concatenate the transformed chunks into a single NumPy array
    transformed_full = np.vstack(transformed_chunks)

    return transformed_full


def parse_dt(df, cols=["timestamp"]):
    return df.assign(
        **{
            col: lambda df: pd.to_datetime(df[col].astype(int), unit="ms")
            for col in cols
        }
    )


def handle_dtypes(df):
    return df.assign(rating=lambda df: df["rating"].astype(float))
