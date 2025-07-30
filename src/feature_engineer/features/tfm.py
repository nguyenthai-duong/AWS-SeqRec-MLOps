import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler


def reshape_2d_to_1d(X):
    """Reshape input array to 1D for compatibility with TF-IDF pipeline.

    Args:
        X: Input array to be reshaped.

    Returns:
        np.ndarray: Reshaped 1D array.
    """
    return np.array(X).reshape(-1)


def flatten_string_array_col(X):
    """Flatten a pandas Series containing lists of strings into single strings.

    Args:
        X (pd.Series): Input Series where each element is a list of strings.

    Returns:
        np.ndarray: Array of flattened strings, with empty values filled as empty strings.

    Raises:
        AssertionError: If input is not a pandas Series or output shape does not match input.
    """
    assert isinstance(X, pd.Series)
    output = X.fillna("").str.join("\n")
    assert X.shape[0] == output.shape[0]
    return output.values


def todense(X):
    """Convert sparse matrix to dense NumPy array.

    Args:
        X: Sparse matrix input.

    Returns:
        np.ndarray: Dense array representation of the input.
    """
    return np.asarray(X.todense())


def title_pipeline_steps():
    """Define preprocessing pipeline steps for title text data.

    Returns:
        list: List of tuples containing pipeline steps for title processing.
              Each tuple contains a step name and a transformer instance.
    """
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="")),
        ("reshape", FunctionTransformer(reshape_2d_to_1d, validate=False)),
        ("tfidf", TfidfVectorizer(min_df=5, max_features=1000, ngram_range=(1, 2))),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps


def description_pipeline_steps():
    """Define preprocessing pipeline steps for description text data.

    Returns:
        list: List of tuples containing pipeline steps for description processing.
              Each tuple contains a step name and a transformer instance.
    """
    steps = [
        (
            "flatten_string_array_col",
            FunctionTransformer(flatten_string_array_col, validate=False),
        ),
        ("tfidf", TfidfVectorizer(min_df=5, max_features=1000, ngram_range=(1, 2))),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps


def tokenizer(s):
    """Tokenize a string by splitting on newlines.

    Args:
        s (str): Input string to tokenize.

    Returns:
        list: List of tokens split by newlines.
    """
    return s.split("\n")


def categories_pipeline_steps():
    """Define preprocessing pipeline steps for category data.

    Returns:
        list: List of tuples containing pipeline steps for category processing.
              Each tuple contains a step name and a transformer instance.
    """
    steps = [
        (
            "flatten_string_array_col",
            FunctionTransformer(flatten_string_array_col, validate=False),
        ),
        ("count_vect", CountVectorizer(tokenizer=tokenizer, token_pattern=None)),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps


def price_parse_dtype(series, pattern):
    """Extract and convert price values from a pandas Series using a regex pattern.

    Args:
        series (pd.Series): Input Series containing price strings.
        pattern (str): Regular expression pattern to extract price values.

    Returns:
        pd.Series: Extracted price values converted to float.
    """
    return series.str.extract(pattern).astype(float)


def price_pipeline_steps(price_pattern=None):
    """Define preprocessing pipeline steps for price data.

    Args:
        price_pattern (str, optional): Regex pattern for price extraction.
                                      Defaults to r"\b((?:\d+\.\d*)|(?:\d+))\b".

    Returns:
        list: List of tuples containing pipeline steps for price processing.
              Each tuple contains a step name and a transformer instance.
    """
    if price_pattern is None:
        price_pattern = r"\b((?:\d+\.\d*)|(?:\d+))\b"
    steps = [
        (
            "extract_price",
            FunctionTransformer(
                price_parse_dtype, kw_args=dict(pattern=price_pattern), validate=False
            ),
        ),
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("min_max_scale", MinMaxScaler()),
    ]
    return steps


def rating_agg_pipeline_steps():
    """Define preprocessing pipeline steps for rating data.

    Returns:
        list: List of tuples containing pipeline steps for rating processing.
              Each tuple contains a step name and a transformer instance.
    """
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("normalize", StandardScaler()),
    ]
    return steps