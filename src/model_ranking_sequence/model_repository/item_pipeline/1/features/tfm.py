import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler


def reshape_2d_to_1d(X):
    """
    Ensure the shape is working for TFIDF pipeline.
    """
    return np.array(X).reshape(-1)


def flatten_string_array_col(X):
    """
    The inputs contain columns with list of sentences. To properly analyze them we would flatten them.
    """
    assert isinstance(X, pd.Series)
    output = X.fillna("").str.join("\n")
    assert X.shape[0] == output.shape[0]
    return output.values


def todense(X):
    return np.asarray(X.todense())


def title_pipeline_steps():
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="")),
        ("reshape", FunctionTransformer(reshape_2d_to_1d, validate=False)),
        ("tfidf", TfidfVectorizer(min_df=5, max_features=1000, ngram_range=(1, 2))),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps


def description_pipeline_steps():
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
    return s.split("\n")


def categories_pipeline_steps():
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
    return series.astype(str).str.extract(pattern).astype(float)


def price_pipeline_steps(price_pattern=None):
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
    steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("normalize", StandardScaler()),
    ]
    return steps
