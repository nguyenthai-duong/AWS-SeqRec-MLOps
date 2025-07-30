import base64
import json
import logging
import os
import traceback
from datetime import datetime, timezone
import pandas as pd
from feast import FeatureStore
from tenacity import retry, stop_after_attempt, wait_fixed


def configure_logging():
    """
    Configures the logging system with a DEBUG level and a standardized format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger()


def initialize_feature_store(repo_path=".", fs_yaml_file="feature_store.yaml"):
    """
    Initializes the Feast FeatureStore with the specified repository path and configuration file.

    Args:
        repo_path (str): Path to the Feast repository. Defaults to current directory.
        fs_yaml_file (str): Name of the Feast configuration YAML file. Defaults to "feature_store.yaml".

    Returns:
        FeatureStore: Initialized Feast FeatureStore instance.
    """
    os.environ["REGISTRY_PATH"] = os.getenv(
        "REGISTRY_PATH",
        "postgresql+psycopg2://postgres:postgres@db-instance-feature-store.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/postgres",
    )
    return FeatureStore(repo_path=repo_path, fs_yaml_file=fs_yaml_file)


logger = configure_logging()
store = initialize_feature_store()
FEATURE_VIEW_NAME = "user_feature_view"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_user_features(user_id):
    """
    Fetches online features for a given user from the Feast FeatureStore with retry logic.

    Args:
        user_id (str): The user ID to fetch features for.

    Returns:
        dict: Dictionary containing the fetched feature vector.

    Raises:
        Exception: If all retry attempts fail to fetch features.
    """
    features = [
        f"{FEATURE_VIEW_NAME}:user_rating_cnt_90d",
        f"{FEATURE_VIEW_NAME}:user_rating_avg_prev_rating_90d",
        f"{FEATURE_VIEW_NAME}:user_rating_list_10_recent_asin",
        f"{FEATURE_VIEW_NAME}:user_rating_list_10_recent_asin_timestamp",
        f"{FEATURE_VIEW_NAME}:item_sequence_ts",
        f"{FEATURE_VIEW_NAME}:item_sequence_ts_bucket",
    ]
    logger.debug(f"[fetch_user_features] user_id={user_id}, features={features}")
    feature_vector = store.get_online_features(
        features=features, entity_rows=[{"user_id": user_id}]
    ).to_dict()
    logger.debug(f"[fetch_user_features] Fetched feature_vector={feature_vector}")
    return feature_vector


def iso_to_unix(ts):
    """
    Converts an ISO timestamp string to a Unix timestamp (seconds since epoch).

    Args:
        ts (str): ISO timestamp string.

    Returns:
        int: Unix timestamp, or -1 if conversion fails.
    """
    try:
        return int(pd.to_datetime(ts).timestamp())
    except Exception:
        return -1


def bucketize(seconds):
    """
    Assigns a time bucket to a time difference (in seconds) based on predefined intervals.

    Args:
        seconds (int): Time difference in seconds.

    Returns:
        int: Time bucket index (0-9) based on the time difference.
    """
    if seconds < 60 * 10:  # 10 minutes
        return 0
    if seconds < 60 * 60:  # 1 hour
        return 1
    if seconds < 60 * 60 * 24:  # 1 day
        return 2
    if seconds < 60 * 60 * 24 * 7:  # 1 week
        return 3
    if seconds < 60 * 60 * 24 * 30:  # 1 month
        return 4
    if seconds < 60 * 60 * 24 * 365:  # 1 year
        return 5
    if seconds < 60 * 60 * 24 * 365 * 3:  # 3 years
        return 6
    if seconds < 60 * 60 * 24 * 365 * 5:  # 5 years
        return 7
    if seconds < 60 * 60 * 24 * 365 * 10:  # 10 years
        return 8
    return 9


def update_and_write_to_online(user_id, asin, rating, timestamp):
    """
    Updates user features with a new rating and writes them to the Feast online store.

    Args:
        user_id (str): The user ID.
        asin (str): The item ASIN.
        rating (float): The rating given by the user.
        timestamp (str): The timestamp of the rating in ISO format.
    """
    logger.debug(
        f"[update_and_write_to_online] user_id={user_id}, asin={asin}, rating={rating}, timestamp={timestamp}"
    )
    fv = store.get_feature_view(FEATURE_VIEW_NAME)
    schema_fields = [f.name for f in fv.schema]
    user_feature = fetch_user_features(user_id)
    logger.debug(f"[update_and_write_to_online] user_feature={user_feature}")

    # Extract and update recent ASINs and timestamps
    old_asins = (
        user_feature.get("user_rating_list_10_recent_asin", [""])[0].split(",")
        if user_feature.get("user_rating_list_10_recent_asin", [""])[0]
        else []
    )
    old_timestamps = (
        user_feature.get("user_rating_list_10_recent_asin_timestamp", [""])[0].split(",")
        if user_feature.get("user_rating_list_10_recent_asin_timestamp", [""])[0]
        else []
    )
    new_asins = (old_asins + [asin])[-10:]
    new_timestamps = (old_timestamps + [timestamp])[-10:]

    # Convert timestamps to Unix format
    item_sequence_ts = [iso_to_unix(ts) for ts in new_timestamps]
    if len(item_sequence_ts) < 10:
        item_sequence_ts = [-1] * (10 - len(item_sequence_ts)) + item_sequence_ts

    # Determine current timestamp
    current_ts = (
        item_sequence_ts[-1]
        if item_sequence_ts and item_sequence_ts[-1] != -1
        else int(datetime.now(timezone.utc).timestamp())
    )

    # Bucketize timestamps
    item_sequence_ts_bucket = [
        -1 if ts == -1 else bucketize(current_ts - ts) for ts in item_sequence_ts
    ]
    if len(item_sequence_ts_bucket) < 10:
        item_sequence_ts_bucket = [-1] * (10 - len(item_sequence_ts_bucket)) + item_sequence_ts_bucket

    # Prepare feature data for online store
    feature_data = {}
    for k in schema_fields:
        if k == "user_rating_list_10_recent_asin":
            feature_data[k] = ",".join(new_asins)
        elif k == "user_rating_list_10_recent_asin_timestamp":
            feature_data[k] = ",".join(new_timestamps)
        elif k == "item_sequence_ts_bucket":
            feature_data[k] = item_sequence_ts_bucket
        elif k == "item_sequence_ts":
            feature_data[k] = item_sequence_ts
        else:
            val = user_feature.get(k, [None])
            feature_data[k] = val[0] if isinstance(val, list) and len(val) == 1 else val

    feature_data["user_id"] = user_id
    feature_data["event_timestamp"] = datetime.now(timezone.utc)
    logger.debug(f"[update_and_write_to_online] Final feature_data={feature_data}")
    store.write_to_online_store(FEATURE_VIEW_NAME, pd.DataFrame([feature_data]))
    logger.info(f"Updated feature for user_id={user_id}")


def lambda_handler(event, context):
    """
    AWS Lambda handler to process Kinesis records and update user features in the Feast online store.

    Args:
        event (dict): AWS Lambda event containing Kinesis records.
        context (object): AWS Lambda context object.

    Returns:
        dict: A dictionary indicating the processing status.

    Raises:
        Exception: Logs and traces any errors during record processing, but continues with remaining records.
    """
    logger.info(f"Received event: {json.dumps(event)[:1000]}")
    for record in event["Records"]:
        try:
            payload = record["kinesis"]["data"]
            logger.debug(f"Payload base64: {payload}")
            data = json.loads(base64.b64decode(payload))
            logger.debug(f"Decoded record: {data}")
            user_id = data["data"]["user_id"]
            parent_asin = data["data"]["parent_asin"]
            rating = data["data"]["rating"]
            timestamp = data["data"]["timestamp"]
            update_and_write_to_online(user_id, parent_asin, rating, timestamp)
        except Exception as ex:
            logger.error(f"Error processing record: {ex}")
            logger.error(traceback.format_exc())
    return {"status": "done"}