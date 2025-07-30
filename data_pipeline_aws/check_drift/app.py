import json
import os
import time
import boto3
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine


def load_configurations():
    """
    Loads configuration settings from environment variables for AWS and application settings.

    Returns:
        tuple: A tuple containing AWS region, access key, secret key, Kinesis stream name,
               polling interval, and minimum messages required for drift check.
    """
    region = os.getenv("AWS_REGION")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    stream_name = os.getenv("STREAM_NAME", "test")
    interval = int(os.getenv("INTERVAL", "1"))
    min_messages = int(os.getenv("MIN_MESSAGES", "3"))
    return region, access_key, secret_key, stream_name, interval, min_messages


def initialize_kinesis_client(region, access_key, secret_key):
    """
    Initializes a boto3 Kinesis client using provided AWS credentials.

    Args:
        region (str): AWS region name.
        access_key (str): AWS access key ID.
        secret_key (str): AWS secret access key.

    Returns:
        boto3.client: Configured Kinesis client.
    """
    return boto3.client(
        "kinesis",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def fix_dtypes(df):
    """
    Standardizes data types for a DataFrame to ensure consistency for drift analysis.

    Args:
        df (pd.DataFrame): Input DataFrame to process.

    Returns:
        pd.DataFrame: Processed DataFrame with standardized column types and no missing values.
    """
    cols_check = ["user_id", "parent_asin", "rating", "timestamp", "main_category", "price"]
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
    return df[cols_check].dropna()


def load_reference_data():
    """
    Loads reference data from either an RDS database or a local Parquet file based on configuration.

    Returns:
        pd.DataFrame: Reference DataFrame with standardized data types.
    """
    use_rds = os.getenv("USE_RDS", "0") == "1"
    if use_rds:
        print("Loading reference data from RDS...")
        username = os.getenv("RDS_USERNAME")
        password = os.getenv("RDS_PASSWORD")
        host = os.getenv("RDS_HOST")
        port = os.getenv("RDS_PORT", "5432")
        database = os.getenv("RDS_DATABASE")
        schema = os.getenv("RDS_SCHEMA", "public")
        table_name = os.getenv("RDS_TABLE", "reviews")
        connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)
        query = f"SELECT * FROM {schema}.{table_name};"
        df_ref = pd.read_sql(query, engine)
    else:
        print("Loading reference data from local file...")
        local_ref_path = os.getenv("LOCAL_REF_PATH", "/home/duong/Documents/datn1/data/to_insert_df.parquet")
        df_ref = pd.read_parquet(local_ref_path)
    
    print("Reference data shape:", df_ref.shape)
    df_ref = fix_dtypes(df_ref)
    print("Reference columns and types:")
    print(df_ref.dtypes)
    print(df_ref.head())
    return df_ref


def get_shard_iterators(kinesis_client, stream_name):
    """
    Retrieves shard iterators for all shards in a Kinesis stream.

    Args:
        kinesis_client (boto3.client): Kinesis client instance.
        stream_name (str): Name of the Kinesis stream.

    Returns:
        list: List of shard iterator IDs.
    """
    resp = kinesis_client.describe_stream(StreamName=stream_name)
    shard_iterators = []
    for shard in resp["StreamDescription"]["Shards"]:
        shard_id = shard["ShardId"]
        shard_iter = kinesis_client.get_shard_iterator(
            StreamName=stream_name, ShardId=shard_id, ShardIteratorType="LATEST"
        )["ShardIterator"]
        shard_iterators.append(shard_iter)
    return shard_iterators


def pull_records(kinesis_client, shard_iterators):
    """
    Pulls records from Kinesis shards and returns the total number of new records,
    the records themselves, and updated shard iterators.

    Args:
        kinesis_client (boto3.client): Kinesis client instance.
        shard_iterators (list): List of shard iterator IDs.

    Returns:
        tuple: (total_new_records, list_of_records, updated_shard_iterators)
    """
    total_new = 0
    next_iterators = []
    all_records = []
    for shard_iterator in shard_iterators:
        response = kinesis_client.get_records(ShardIterator=shard_iterator, Limit=100)
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


def preprocess_df_current(cached_records):
    """
    Preprocesses raw Kinesis records into a DataFrame with standardized data types.

    Args:
        cached_records (list): List of raw Kinesis records as strings.

    Returns:
        pd.DataFrame: Processed DataFrame with standardized types, or None if no valid data.
    """
    rows = []
    for msg in cached_records:
        try:
            row = json.loads(msg)["data"]
            rows.append(row)
        except Exception as ex:
            print(f"JSON decode error: {ex}")
    if not rows:
        return None
    df_current = pd.DataFrame(rows)
    df_current = fix_dtypes(df_current)
    print("Current data shape:", df_current.shape)
    print("Current columns and types:")
    print(df_current.dtypes)
    print(df_current.head())
    return df_current


def log_drift_details(report):
    """
    Logs drift details (score, p-value, statistical test, and detection status) for each column.

    Args:
        report (evidently.report.Report): Evidently drift report object.
    """
    try:
        feature_metrics = report.as_dict()["metrics"][0]["result"]["drift_by_columns"]
        for col, metrics in feature_metrics.items():
            score = metrics.get("drift_score")
            stattest = metrics.get("stattest_name")
            actual_pvalue = metrics.get("stattest_p_value")
            detected = metrics.get("drift_detected")
            print(
                f"[{col}] | Drift Score: {score:.4f} | p-value: {actual_pvalue:.4g} | "
                f"StatTest: {stattest} | Drift Detected: {detected}"
            )
    except Exception as ex:
        print(f"Failed to log drift details: {ex}")


def upload_html_to_s3(local_path, bucket, s3_key, region, access_key, secret_key):
    """
    Uploads an HTML report to an S3 bucket.

    Args:
        local_path (str): Local path to the HTML file.
        bucket (str): S3 bucket name.
        s3_key (str): S3 key for the uploaded file.
        region (str): AWS region name.
        access_key (str): AWS access key ID.
        secret_key (str): AWS secret access key.
    """
    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    s3.upload_file(local_path, bucket, s3_key, ExtraArgs={"ContentType": "text/html"})
    print(f"Uploaded to s3://{bucket}/{s3_key}")


def main():
    """
    Main loop for monitoring data drift in Kinesis stream data.
    Periodically pulls records, performs drift analysis against reference data,
    logs results, and uploads HTML reports to S3.
    """
    region, access_key, secret_key, stream_name, interval, min_messages = load_configurations()
    kinesis_client = initialize_kinesis_client(region, access_key, secret_key)
    df_ref = load_reference_data()
    shard_iterators = get_shard_iterators(kinesis_client, stream_name)
    print(f"Listening to {len(shard_iterators)} shard(s) in stream: {stream_name}")

    s3_bucket = os.getenv("S3_BUCKET", "recsys-ops")
    s3_key = os.getenv("S3_KEY", "drift_report.html")

    while True:
        print(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for at least {min_messages} "
            f"new messages to check drift ..."
        )
        count_since_last_pull = 0
        cached_records = []
        while count_since_last_pull < min_messages:
            time.sleep(interval)
            total_new, all_records, shard_iterators = pull_records(kinesis_client, shard_iterators)
            count_since_last_pull += total_new
            cached_records.extend(all_records)
            print(
                f"Pulled {total_new} new message(s), total since last pull: {count_since_last_pull}"
            )

        print(f"\n------ Drift check triggered: {count_since_last_pull} message(s) ------")
        df_current = preprocess_df_current(cached_records)
        if df_current is None or df_current.empty:
            print("No valid current data, skip drift check.")
            continue

        # Validate columns and data types
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

        print("Running Evidently report...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_ref, current_data=df_current)
        log_drift_details(report)
        report_path = f"drift_report_{int(time.time())}.html"
        report.save_html(report_path)
        print(f"Successfully exported report to file: {report_path}")
        upload_html_to_s3(report_path, s3_bucket, s3_key, region, access_key, secret_key)


if __name__ == "__main__":
    main()