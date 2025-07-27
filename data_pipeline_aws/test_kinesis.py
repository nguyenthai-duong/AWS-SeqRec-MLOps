import json
import os
import time

import boto3
import pandas as pd
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine

# ---- 1. Load AWS credentials từ ../.env ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))
REGION = os.getenv("AWS_REGION")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# ---- 2. Kinesis config ----
STREAM_NAME = "test"
INTERVAL = 1  # seconds
MIN_MESSAGES = 3

kinesis = boto3.client(
    "kinesis",
    region_name=REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# ---- 3. Load reference data ----
USE_RDS = os.getenv("USE_RDS", "0") == "1"
cols_check = ["user_id", "parent_asin", "rating", "timestamp", "main_category", "price"]


def fix_dtypes(df):
    df = df.copy()
    for col in cols_check:
        if col not in df.columns:
            df[col] = None
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["main_category"] = df["main_category"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df[cols_check].dropna()
    return df


if USE_RDS:
    print("Đang load reference data từ RDS...")
    username = "postgres"
    password = "postgres"
    host = "simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com"
    port = "5432"
    database = "raw_data"
    schema = "public"
    table_name = "reviews"
    connection_string = (
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    )
    engine = create_engine(connection_string)
    query = f"SELECT * FROM {schema}.{table_name};"
    df_ref = pd.read_sql(query, engine)
else:
    print("Đang load reference data từ file local...")
    df_ref = pd.read_parquet("/home/duong/Documents/datn1/data/to_insert_df.parquet")
print("Reference data shape:", df_ref.shape)
df_ref = fix_dtypes(df_ref)
print("Reference columns and types:")
print(df_ref.dtypes)
print(df_ref.head())


# ---- 4. Hàm lấy shard iterator cho các shard ----
def get_shard_iterators(stream_name):
    resp = kinesis.describe_stream(StreamName=stream_name)
    shard_iterators = []
    for shard in resp["StreamDescription"]["Shards"]:
        shard_id = shard["ShardId"]
        shard_iter = kinesis.get_shard_iterator(
            StreamName=stream_name, ShardId=shard_id, ShardIteratorType="LATEST"
        )["ShardIterator"]
        shard_iterators.append(shard_iter)
    return shard_iterators


# ---- 5. Hàm pull records từ Kinesis ----
def pull_records(shard_iterators):
    total_new = 0
    next_iterators = []
    all_records = []
    for i, shard_iterator in enumerate(shard_iterators):
        response = kinesis.get_records(ShardIterator=shard_iterator, Limit=100)
        records = response["Records"]
        next_iterators.append(response["NextShardIterator"])
        if records:
            for record in records:
                try:
                    all_records.append(record["Data"].decode("utf-8"))
                except Exception as ex:
                    print(f"Decode error: {ex}")
            total_new += len(records)
    return total_new, all_records, next_iterators


# ---- 6. Hàm chuẩn hóa dataframe cho current data ----
def preprocess_df_current(cached_records):
    rows = []
    for msg in cached_records:
        try:
            row = json.loads(msg)["data"]
            rows.append(row)
        except Exception as ex:
            print(f"JSON decode error: {ex}")
    df_current = pd.DataFrame(rows)
    df_current = fix_dtypes(df_current)
    print("Current data shape:", df_current.shape)
    print("Current columns and types:")
    print(df_current.dtypes)
    print(df_current.head())
    return df_current


# ---- 7. Hàm log p-value và drift score từng cột ----
def log_drift_details(report):
    # Trích drift score và p-value từ output Evidently
    result = report.as_dict()
    try:
        feature_metrics = result["metrics"][0]["result"]["drift_by_columns"]
        for col, metrics in feature_metrics.items():
            score = metrics.get("drift_score")
            stattest = metrics.get("stattest_name")
            actual_pvalue = metrics.get("stattest_p_value")
            detected = metrics.get("drift_detected")
            print(
                f"[{col}] | Drift Score: {score:.4f} | p-value: {actual_pvalue:.4g} | StatTest: {stattest} | Drift Detected: {detected}"
            )
    except Exception as ex:
        print("Không log được chi tiết drift:", ex)


def upload_html_to_s3(local_path, bucket="recsys-ops", s3_key="drift_report.html"):
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    s3.upload_file(local_path, bucket, s3_key, ExtraArgs={"ContentType": "text/html"})
    print(f"Đã upload lên s3://{bucket}/{s3_key}")


# ---- 8. Main loop ----
def main():
    shard_iterators = get_shard_iterators(STREAM_NAME)
    print(f"Listening to {len(shard_iterators)} shard(s) in stream: {STREAM_NAME}")

    while True:
        print(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for at least {MIN_MESSAGES} new messages to check drift ..."
        )
        count_since_last_pull = 0
        cached_records = []
        while count_since_last_pull < MIN_MESSAGES:
            time.sleep(INTERVAL)
            total_new, all_records, shard_iterators = pull_records(shard_iterators)
            count_since_last_pull += total_new
            cached_records.extend(all_records)
            print(
                f"Pulled {total_new} new message(s), total since last pull: {count_since_last_pull}"
            )

        print(
            f"\n------ Drift check triggered: {count_since_last_pull} message(s) ------"
        )
        df_current = preprocess_df_current(cached_records)
        if df_current is None or df_current.empty:
            print("No valid current data, skip drift check.")
            continue

        # Kiểm tra cột và dtype lần cuối
        if not all(df_ref.columns == df_current.columns):
            print("Column mismatch between reference and current data, skip.")
            print("df_ref.columns:", df_ref.columns)
            print("df_current.columns:", df_current.columns)
            continue
        if not all(df_ref.dtypes == df_current.dtypes):
            print("Dtype mismatch between reference and current data, skip.")
            print("df_ref.dtypes:", df_ref.dtypes)
            print("df_current.dtypes:", df_current.dtypes)
            continue

        print("Đang chạy Evidently report...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_ref, current_data=df_current)
        # Log chi tiết drift
        log_drift_details(report)
        # Save HTML report
        report_path = f"drift_report_{int(time.time())}.html"
        report.save_html(report_path)
        print(f"Xuất báo cáo thành công ra file: {report_path}")
        upload_html_to_s3(report_path, bucket="recsys-ops", s3_key="drift_report.html")


if __name__ == "__main__":
    main()
