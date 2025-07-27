from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine


def get_engine():
    return create_engine(
        "postgresql://postgres:postgres@simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/raw_data"
    )


def stringify_pg_array(x):
    if isinstance(x, (list, np.ndarray, set)):
        return "{" + ", ".join(f'"{str(item)}"' for item in x) + "}"
    return str(x)


def insert_one_row():
    parquet_path = Path(__file__).resolve().parents[1] / "data" / "holdout.parquet"
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)

    current_index = int(Variable.get("current_index", default_var=0))
    if current_index >= total_rows:
        print("All rows inserted.")
        return

    row_df = df.iloc[[current_index]]

    # Convert mảng thành string nếu cần
    for col in ["description", "categories"]:
        if col in row_df.columns:
            row_df[col] = row_df[col].apply(stringify_pg_array)

    # Log row sắp insert
    print(f"Row {current_index} to insert:")
    print(row_df.to_markdown(index=False))

    engine = get_engine()
    with engine.begin() as conn:
        row_df.to_sql(
            "new_reviews",
            conn,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
        )
    print(f"Inserted row {current_index}")

    Variable.set("current_index", current_index + 1)


with DAG(
    dag_id="insert_next_row_dag",
    start_date=datetime(2025, 4, 5),
    schedule_interval=None,
    catchup=False,
    description="Insert exactly 1 row (after manual preview)",
) as dag:
    insert_task = PythonOperator(
        task_id="insert_row",
        python_callable=insert_one_row,
    )
