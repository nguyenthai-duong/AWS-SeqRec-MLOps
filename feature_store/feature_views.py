from feast import FeatureView, Field
from feast.types import Int64, Float32, String, Array
from datetime import timedelta
from entities import user, parent_asin
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource


user_feature_source = SparkSource(
    path="s3a://recsys-ops/feature-store/user_rating_stats/",
    timestamp_field="event_timestamp",
    name="user_feature_source",
    file_format="parquet",
)

parent_feature_source = SparkSource(
    path="s3a://recsys-ops/feature-store/parent_asin_rating_stats/",
    timestamp_field="event_timestamp",
    name="parent_feature_source",
    file_format="parquet",
)

train_source = SparkSource(
    path="s3a://recsys-ops/feature-store/train/train.parquet/",
    timestamp_field="timestamp",
    name="train_source",
    file_format="parquet",
)

val_source = SparkSource(
    path="s3a://recsys-ops/feature-store/val/val.parquet/",
    timestamp_field="timestamp",
    name="val_source",
    file_format="parquet",
)

user_feature_view = FeatureView(
    name="user_feature_view",
    entities=[user],
    ttl=timedelta(days=10000),
    schema=[
        Field(name="user_rating_cnt_90d", dtype=Int64),
        Field(name="user_rating_avg_prev_rating_90d", dtype=Float32),
        Field(name="user_rating_list_10_recent_asin", dtype=String),
        Field(name="user_rating_list_10_recent_asin_timestamp", dtype=String),
        Field(name="item_sequence_ts", dtype=Array(Int64)),
        Field(name="item_sequence_ts_bucket", dtype=Array(Int64))
    ],
    source=user_feature_source,
    online=True
)

parent_feature_view = FeatureView(
    name="parent_asin_feature_view",
    entities=[parent_asin],
    ttl=timedelta(days=10000),
    schema=[
        Field(name="parent_asin_rating_cnt_365d", dtype=Int64),
        Field(name="parent_asin_rating_avg_prev_rating_365d", dtype=Float32),
        Field(name="parent_asin_rating_cnt_90d", dtype=Int64),
        Field(name="parent_asin_rating_avg_prev_rating_90d", dtype=Float32),
        Field(name="parent_asin_rating_cnt_30d", dtype=Int64),
        Field(name="parent_asin_rating_avg_prev_rating_30d", dtype=Float32),
        Field(name="parent_asin_rating_cnt_7d", dtype=Int64),
        Field(name="parent_asin_rating_avg_prev_rating_7d", dtype=Float32),
        Field(name="main_category", dtype=String),
        Field(name="categories", dtype=Array(String)),
        Field(name="price", dtype=String)
    ],
    source=parent_feature_source,
    online=True
)

train_feature_view = FeatureView(
    name="train_feature_view",
    entities=[user, parent_asin],
    ttl=timedelta(days=10000),
    schema=[
        Field(name="user_id", dtype=String),
        Field(name="parent_asin", dtype=String),
        Field(name="rating", dtype=Int64),
        Field(name="timestamp", dtype=String)
    ],
    source=train_source,
    online=False
)

val_feature_view = FeatureView(
    name="val_feature_view",
    entities=[user, parent_asin],
    ttl=timedelta(days=10000),
    schema=[
        Field(name="user_id", dtype=String),
        Field(name="parent_asin", dtype=String),
        Field(name="rating", dtype=Int64),
        Field(name="timestamp", dtype=String)
    ],
    source=val_source,
    online=False
)
