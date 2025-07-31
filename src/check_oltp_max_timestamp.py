import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

# PostgreSQL connection details
schema = "public"
table_name = "reviews"
# Create a connection string and engine outside the function
connection_string = "postgresql://postgres:postgres@simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/raw_data"
engine = create_engine(connection_string)


def get_curr_oltp_max_timestamp():
    query = f"SELECT max(timestamp) as max_timestamp FROM {schema}.{table_name};"
    max_timestamp = pd.read_sql(query, engine)["max_timestamp"].iloc[0]

    # Convert the timestamp to a timezone-aware format (UTC +00:00)
    if pd.notnull(max_timestamp):
        max_timestamp = pd.to_datetime(max_timestamp).tz_localize("UTC")

    return max_timestamp


logger.info(f"Max timestamp in OLTP: <ts>{get_curr_oltp_max_timestamp()}</ts>")
