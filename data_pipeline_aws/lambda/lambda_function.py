import base64
import json
import logging
import os
import traceback
from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore
from tenacity import retry, stop_after_attempt, wait_fixed

# --- Logging setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

# --- Feast config ---
FEAST_REPO_PATH = "."
FS_YAML_FILE = "feature_store.yaml"
FEATURE_VIEW_NAME = "user_feature_view"

# Set registry path from environment variable
os.environ["REGISTRY_PATH"] = os.getenv(
    "REGISTRY_PATH",
    "postgresql+psycopg2://postgres:postgres@db-instance-feature-store.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/postgres",
)
store = FeatureStore(repo_path=FEAST_REPO_PATH, fs_yaml_file=FS_YAML_FILE)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_user_features(user_id):
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


def update_and_write_to_online(user_id, asin, rating, timestamp):
    logger.debug(
        f"[update_and_write_to_online] user_id={user_id}, asin={asin}, rating={rating}, timestamp={timestamp}"
    )
    fv = store.get_feature_view(FEATURE_VIEW_NAME)
    schema_fields = [f.name for f in fv.schema]
    user_feature = fetch_user_features(user_id)
    logger.debug(f"[update_and_write_to_online] user_feature={user_feature}")

    old_asins = (
        user_feature.get("user_rating_list_10_recent_asin", [""])[0].split(",")
        if user_feature.get("user_rating_list_10_recent_asin", [""])[0]
        else []
    )
    old_timestamps = (
        user_feature.get("user_rating_list_10_recent_asin_timestamp", [""])[0].split(
            ","
        )
        if user_feature.get("user_rating_list_10_recent_asin_timestamp", [""])[0]
        else []
    )
    new_asins = (old_asins + [asin])[-10:]
    new_timestamps = (old_timestamps + [timestamp])[-10:]

    def iso_to_unix(ts):
        try:
            return int(pd.to_datetime(ts).timestamp())
        except Exception:
            return -1

    item_sequence_ts = [iso_to_unix(ts) for ts in new_timestamps]
    if len(item_sequence_ts) < 10:
        item_sequence_ts = [-1] * (10 - len(item_sequence_ts)) + item_sequence_ts

    current_ts = (
        item_sequence_ts[-1]
        if item_sequence_ts and item_sequence_ts[-1] != -1
        else int(datetime.now(timezone.utc).timestamp())
    )

    def bucketize(seconds):
        if seconds < 60 * 10:
            return 0
        if seconds < 60 * 60:
            return 1
        if seconds < 60 * 60 * 24:
            return 2
        if seconds < 60 * 60 * 24 * 7:
            return 3
        if seconds < 60 * 60 * 24 * 30:
            return 4
        if seconds < 60 * 60 * 24 * 365:
            return 5
        if seconds < 60 * 60 * 24 * 365 * 3:
            return 6
        if seconds < 60 * 60 * 24 * 365 * 5:
            return 7
        if seconds < 60 * 60 * 24 * 365 * 10:
            return 8
        return 9

    item_sequence_ts_bucket = [
        -1 if ts == -1 else bucketize(current_ts - ts) for ts in item_sequence_ts
    ]
    if len(item_sequence_ts_bucket) < 10:
        item_sequence_ts_bucket = [-1] * (
            10 - len(item_sequence_ts_bucket)
        ) + item_sequence_ts_bucket

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
            if isinstance(val, list) and len(val) == 1:
                feature_data[k] = val[0]
            else:
                feature_data[k] = val

    feature_data["user_id"] = user_id
    feature_data["event_timestamp"] = datetime.now(timezone.utc)
    logger.debug(f"[update_and_write_to_online] Final feature_data={feature_data}")
    store.write_to_online_store(FEATURE_VIEW_NAME, pd.DataFrame([feature_data]))
    logger.info(f"Updated feature for user_id={user_id}")


def lambda_handler(event, context):
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
