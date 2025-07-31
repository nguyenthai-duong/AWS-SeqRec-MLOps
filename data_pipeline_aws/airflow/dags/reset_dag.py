import os
from datetime import datetime

import boto3
import dotenv
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine


def get_engine():
    """
    Creates and returns a SQLAlchemy engine for connecting to the PostgreSQL database using Airflow Variables.

    Returns:
        sqlalchemy.engine.Engine: A SQLAlchemy engine instance connected to the raw_data database.
    """
    db_user = Variable.get("db_user", default_var="postgres")
    db_password = Variable.get("db_password", default_var="postgres")
    db_host = Variable.get(
        "db_host",
        default_var="simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com",
    )
    db_port = Variable.get("db_port", default_var="5432")
    db_name = Variable.get("db_name", default_var="raw_data")

    connection_string = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    return create_engine(connection_string)


def reset_and_pull():
    """
    Performs three main tasks:
    1. Truncates the new_reviews table in the PostgreSQL database.
    2. Resets the Airflow Variable 'current_index' to 0.
    3. Downloads the holdout.parquet file from an S3 bucket to a local path.

    Raises:
        Exception: If any step (truncate, reset, or download) fails, an exception is raised with an error message.
    """
    # 1. Truncate the new_reviews table
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute("TRUNCATE TABLE public.new_reviews;")
        print("Successfully truncated the new_reviews table!")
    except Exception as e:
        print(f"Error truncating table new_reviews: {e}")
        raise

    # 2. Reset Airflow Variable
    try:
        Variable.set("current_index", 0)
        print("current_index reset to 0")
    except Exception as e:
        print(f"Error resetting current_index: {e}")
        raise

    # 3. Download file from S3 to local
    try:
        dotenv.load_dotenv("/opt/airflow/.env", override=True)

        s3_bucket = os.getenv("S3_BUCKET")
        s3_key = os.getenv("S3_KEY", "holdout.parquet")
        s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        s3_region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")

        if not all([s3_bucket, s3_access_key, s3_secret_key]):
            raise ValueError("Missing required AWS credentials or S3 bucket name")

        local_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "holdout.parquet")
        )
        print(f"Will pull s3://{s3_bucket}/{s3_key} -> {local_path}")

        s3 = boto3.client(
            "s3",
            region_name=s3_region,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(s3_bucket, s3_key, local_path)
        print("Successfully downloaded holdout.parquet from S3!")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        raise


with DAG(
    dag_id="reset_index_and_pull_holdout",
    start_date=datetime(2025, 4, 5),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    description="Resets the current_index, truncates the new_reviews table, and downloads the latest holdout.parquet file from S3",
) as dag:
    reset_and_pull_task = PythonOperator(
        task_id="reset_and_pull",
        python_callable=reset_and_pull,
    )
