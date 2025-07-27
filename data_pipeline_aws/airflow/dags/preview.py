from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator


def preview_next_row():
    try:
        parquet_path = Path(__file__).resolve().parents[1] / "data" / "holdout.parquet"
        print(f"Reading from: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        current_index = int(Variable.get("current_index", default_var=0))
        print(f"Current index: {current_index} / Total rows: {len(df)}")

        if current_index >= len(df):
            print("All rows already inserted.")
            return

        row_df = df.iloc[[current_index]]
        print("Previewing row:")
        print(row_df.to_markdown(index=False))

    except Exception as e:
        print("Error in preview_next_row:")
        import traceback

        traceback.print_exc()
        raise e


with DAG(
    dag_id="preview_next_row_dag",
    start_date=datetime(2025, 4, 5),
    schedule_interval=None,
    catchup=False,
    description="Preview 1 row before inserting (manual approval)",
) as dag:
    preview_task = PythonOperator(
        task_id="preview_row",
        python_callable=preview_next_row,
    )
