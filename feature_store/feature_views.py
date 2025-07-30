from feast import FeatureView, Field
from feast.types import Int64, Float32, String, Array
from datetime import timedelta
from entities import user, parent_asin
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource


def create_user_feature_source():
    """
    Creates a SparkSource for user feature data stored in S3.

    Returns:
        SparkSource: Configured SparkSource for user feature data.
    """
    return SparkSource(
        path="s3a://recsys-ops/feature-store/user_rating_stats/",
        timestamp_field="event_timestamp",
        name="user_feature_source",
        file_format="parquet",
    )


def create_parent_feature_source():
    """
    Creates a SparkSource for parent ASIN feature data stored in S3.

    Returns:
        SparkSource: Configured SparkSource for parent ASIN feature data.
    """
    return SparkSource(
        path="s3a://recsys-ops/feature-store/parent_asin_rating_stats/",
        timestamp_field="event_timestamp",
        name="parent_feature_source",
        file_format="parquet",
    )


def create_train_source():
    """
    Creates a SparkSource for training data stored in S3.

    Returns:
        SparkSource: Configured SparkSource for training data.
    """
    return SparkSource(
        path="s3a://recsys-ops/feature-store/train/train.parquet/",
        timestamp_field="timestamp",
        name="train_source",
        file_format="parquet",
    )


def create_val_source():
    """
    Creates a SparkSource for validation data stored in S3.

    Returns:
        SparkSource: Configured SparkSource for validation data.
    """
    return SparkSource(
        path="s3a://recsys-ops/feature-store/val/val.parquet/",
        timestamp_field="timestamp",
        name="val_source",
        file_format="parquet",
    )


def create_user_feature_view():
    """
    Defines a FeatureView for user-related features.

    Returns:
        FeatureView: Configured FeatureView for user features with online serving enabled.
    """
    return FeatureView(
        name="user_feature_view",
        entities=[user],
        ttl=timedelta(days=10000),
        schema=[
            Field(name="user_rating_cnt_90d", dtype=Int64, description="Number of ratings by the user in the last 90 days."),
            Field(name="user_rating_avg_prev_rating_90d", dtype=Float32, description="Average rating by the user in the last 90 days."),
            Field(name="user_rating_list_10_recent_asin", dtype=String, description="Comma-separated list of 10 most recent ASINs rated by the user."),
            Field(name="user_rating_list_10_recent_asin_timestamp", dtype=String, description="Comma-separated list of timestamps for the 10 most recent ASINs rated."),
            Field(name="item_sequence_ts", dtype=Array(Int64), description="Unix timestamps for the 10 most recent user ratings."),
            Field(name="item_sequence_ts_bucket", dtype=Array(Int64), description="Time bucket indices for the 10 most recent user ratings.")
        ],
        source=create_user_feature_source(),
        online=True
    )


def create_parent_feature_view():
    """
    Defines a FeatureView for parent ASIN-related features.

    Returns:
        FeatureView: Configured FeatureView for parent ASIN features with online serving enabled.
    """
    return FeatureView(
        name="parent_asin_feature_view",
        entities=[parent_asin],
        ttl=timedelta(days=10000),
        schema=[
            Field(name="parent_asin_rating_cnt_365d", dtype=Int64, description="Number of ratings for the parent ASIN in the last 365 days."),
            Field(name="parent_asin_rating_avg_prev_rating_365d", dtype=Float32, description="Average rating for the parent ASIN in the last 365 days."),
            Field(name="parent_asin_rating_cnt_90d", dtype=Int64, description="Number of ratings for the parent ASIN in the last 90 days."),
            Field(name="parent_asin_rating_avg_prev_rating_90d", dtype=Float32, description="Average rating for the parent ASIN in the last 90 days."),
            Field(name="parent_asin_rating_cnt_30d", dtype=Int64, description="Number of ratings for the parent ASIN in the last 30 days."),
            Field(name="parent_asin_rating_avg_prev_rating_30d", dtype=Float32, description="Average rating for the parent ASIN in the last 30 days."),
            Field(name="parent_asin_rating_cnt_7d", dtype=Int64, description="Number of ratings for the parent ASIN in the last 7 days."),
            Field(name="parent_asin_rating_avg_prev_rating_7d", dtype=Float32, description="Average rating for the parent ASIN in the last 7 days."),
            Field(name="main_category", dtype=String, description="Main category of the parent ASIN."),
            Field(name="categories", dtype=Array(String), description="List of categories associated with the parent ASIN."),
            Field(name="price", dtype=String, description="Price of the parent ASIN.")
        ],
        source=create_parent_feature_source(),
        online=True
    )


def create_train_feature_view():
    """
    Defines a FeatureView for training data.

    Returns:
        FeatureView: Configured FeatureView for training data with online serving disabled.
    """
    return FeatureView(
        name="train_feature_view",
        entities=[user, parent_asin],
        ttl=timedelta(days=10000),
        schema=[
            Field(name="user_id", dtype=String, description="Unique identifier for a user."),
            Field(name="parent_asin", dtype=String, description="Unique identifier for a parent ASIN."),
            Field(name="rating", dtype=Int64, description="Rating given by the user to the parent ASIN."),
            Field(name="timestamp", dtype=String, description="Timestamp of the rating event.")
        ],
        source=create_train_source(),
        online=False
    )


def create_val_feature_view():
    """
    Defines a FeatureView for validation data.

    Returns:
        FeatureView: Configured FeatureView for validation data with online serving disabled.
    """
    return FeatureView(
        name="val_feature_view",
        entities=[user, parent_asin],
        ttl=timedelta(days=10000),
        schema=[
            Field(name="user_id", dtype=String, description="Unique identifier for a user."),
            Field(name="parent_asin", dtype=String, description="Unique identifier for a parent ASIN."),
            Field(name="rating", dtype=Int64, description="Rating given by the user to the parent ASIN."),
            Field(name="timestamp", dtype=String, description="Timestamp of the rating event.")
        ],
        source=create_val_source(),
        online=False
    )


user_feature_view = create_user_feature_view()
parent_feature_view = create_parent_feature_view()
train_feature_view = create_train_feature_view()
val_feature_view = create_val_feature_view()