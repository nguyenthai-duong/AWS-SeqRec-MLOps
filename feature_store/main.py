from fastapi import FastAPI
from pydantic import BaseModel
from feast import FeatureStore

store = FeatureStore(
    repo_path=".", fs_yaml_file="feature_store.yaml"
)

app = FastAPI()

class ParentAsinInput(BaseModel):
    parent_asin: str

class UserInput(BaseModel):
    user_id: str

@app.post("/features_parent_asin")
def get_features(input: ParentAsinInput):
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
def get_features(input: UserInput):
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

