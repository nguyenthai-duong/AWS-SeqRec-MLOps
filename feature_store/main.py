from fastapi import FastAPI
from pydantic import BaseModel
from feast import FeatureStore


def initialize_feature_store(repo_path=".", fs_yaml_file="feature_store.yaml"):
    """
    Initializes the Feast FeatureStore with the specified repository path and configuration file.

    Args:
        repo_path (str): Path to the Feast repository. Defaults to the current directory.
        fs_yaml_file (str): Name of the Feast configuration YAML file. Defaults to "feature_store.yaml".

    Returns:
        FeatureStore: Initialized Feast FeatureStore instance.
    """
    return FeatureStore(repo_path=repo_path, fs_yaml_file=fs_yaml_file)


app = FastAPI()
store = initialize_feature_store()


class ParentAsinInput(BaseModel):
    """Pydantic model for validating parent ASIN input."""
    parent_asin: str


class UserInput(BaseModel):
    """Pydantic model for validating user ID input."""
    user_id: str


@app.post("/features_parent_asin")
def get_features_parent_asin(input: ParentAsinInput):
    """
    Retrieves online features for a given parent ASIN from the Feast FeatureStore.

    Args:
        input (ParentAsinInput): Input model containing the parent ASIN.

    Returns:
        dict: Dictionary containing the requested parent ASIN features.

    Raises:
        Exception: If feature retrieval from the Feast FeatureStore fails.
    """
    entity_rows = [{"parent_asin": input.parent_asin}]
    features = [
        "parent_asin_feature_view:parent_asin_rating_cnt_365d",
        "parent_asin_feature_view:parent_asin_rating_avg_prev_rating_365d",
        "parent_asin_feature_view:parent_asin_rating_cnt_90d",
        "parent_asin_feature_view:parent_asin_rating_avg_prev_rating_90d",
        "parent_asin_feature_view:parent_asin_rating_cnt_30d",
        "parent_asin_feature_view:parent_asin_rating_avg_prev_rating_30d",
        "parent_asin_feature_view:parent_asin_rating_cnt_7d",
        "parent_asin_feature_view:parent_asin_rating_avg_prev_rating_7d",
        "parent_asin_feature_view:main_category",
        "parent_asin_feature_view:categories",
        "parent_asin_feature_view:price",
    ]
    result = store.get_online_features(
        features=features,
        entity_rows=entity_rows
    ).to_dict()
    return result


@app.post("/user_features")
def get_features_user(input: UserInput):
    """
    Retrieves online features for a given user ID from the Feast FeatureStore.

    Args:
        input (UserInput): Input model containing the user ID.

    Returns:
        dict: Dictionary containing the requested user features.

    Raises:
        Exception: If feature retrieval from the Feast FeatureStore fails.
    """
    entity_rows = [{"user_id": input.user_id}]
    features = [
        "user_feature_view:user_rating_list_10_recent_asin",
        "user_feature_view:user_rating_list_10_recent_asin_timestamp",
        "user_feature_view:item_sequence_ts_bucket",
    ]
    result = store.get_online_features(
        features=features,
        entity_rows=entity_rows
    ).to_dict()
    return result