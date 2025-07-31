from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine


def get_engine():
    """
    Creates and returns a SQLAlchemy engine for connecting to the PostgreSQL database.

    Returns:
        sqlalchemy.engine.Engine: A SQLAlchemy engine instance connected to the raw_data database.
    """
    return create_engine(
        "postgresql://postgres:postgres@simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/raw_data"
    )


def stringify_pg_array(x):
    """
    Converts a list, NumPy array, or set into a PostgreSQL-compatible array string.

    Args:
        x: Input data, which can be a list, NumPy array, set, or other type.

    Returns:
        str: A PostgreSQL-compatible string representation of the input.
             For lists/arrays/sets, it returns a string in the format '{item1, item2, ...}'.
             For other types, it returns the string representation of the input.
    """
    if isinstance(x, (list, np.ndarray, set)):
        return "{" + ", ".join(f'"{str(item)}"' for item in x) + "}"
    return str(x)


def insert_one_row():
    """
    Inserts a single row from a Parquet file into the new_reviews table in the PostgreSQL database.
    The row is selected based on the current_index stored in Airflow Variables.
    If the current_index exceeds the number of rows in the Parquet file, the function terminates early.
    Arrays in specified columns ('description', 'categories') are converted to PostgreSQL-compatible strings.

    Raises:
        Exception: If an error occurs during reading the Parquet file, database insertion, or variable update.
    """
    parquet_path = Path(__file__).resolve().parents[1] / "data" / "holdout.parquet"
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)

    current_index = int(Variable.get("current_index", default_var=0))
    if current_index >= total_rows:
        print("All rows inserted.")
        return

    row_df = df.iloc[[current_index]]

    # Convert arrays to PostgreSQL-compatible strings
    for col in ["description", "categories"]:
        if col in row_df.columns:
            row_df[col] = row_df[col].apply(stringify_pg_array)

    # Log the row to be inserted
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


start_date_str = Variable.get("dag_start_date", default_var="2025-04-05")
dynamic_start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

with DAG(
    dag_id="insert_next_row_dag",
    start_date=dynamic_start_date,
    schedule_interval=None,
    catchup=False,
    description="Inserts exactly one row from the holdout.parquet file into the new_reviews table after manual preview",
) as dag:
    insert_task = PythonOperator(
        task_id="insert_row",
        python_callable=insert_one_row,
    )
