import json
import logging
import sys
from datetime import datetime

import boto3
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsgluedq.transforms import EvaluateDataQuality
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve Glue Job Parameters
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "S3_BUCKET",
        "DB_CONNECTION",
        "TABLE_NAME",
        "AWS_REGION",
        "DB_USERNAME",
        "DB_PASSWORD",
    ],
)

# Assign parameters to variables
JOB_NAME = args["JOB_NAME"]
S3_BUCKET = args["S3_BUCKET"]
DB_CONNECTION = args["DB_CONNECTION"]
TABLE_NAME = args["TABLE_NAME"]
AWS_REGION = args["AWS_REGION"]
DB_USERNAME = args["DB_USERNAME"]
DB_PASSWORD = args["DB_PASSWORD"]

# Initialize Spark session with optimized configurations
spark = (
    SparkSession.builder.appName("FeastFeatureTransform")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled", "true")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.network.timeout", "600s")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

glueContext = GlueContext(spark.sparkContext)
job = Job(glueContext)
job.init(JOB_NAME, args)


# HELPER FUNCTIONS
def convert_categories(categories):
    """Convert categories string/set to list."""
    if not categories:
        return []

    if isinstance(categories, str):
        if categories.startswith("{") and categories.endswith("}"):
            cleaned = categories.strip("{}")
            if cleaned:
                items = [item.strip().strip('"') for item in cleaned.split(",")]
                return items
            return []
        try:
            parsed = json.loads(categories)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, (set, tuple)):
                return list(parsed)
            return [categories]
        except json.JSONDecodeError:
            return [categories]
    elif isinstance(categories, (set, tuple)):
        return list(categories)
    elif isinstance(categories, list):
        return categories
    return [str(categories)]


convert_categories_udf = F.udf(convert_categories, ArrayType(StringType()))


# LOAD DATA
def load_data():
    """Load data from PostgreSQL with error handling."""
    try:
        dynamic_frame = glueContext.create_dynamic_frame.from_options(
            connection_type="postgresql",
            connection_options={
                "url": DB_CONNECTION,
                "dbtable": TABLE_NAME,
                "user": DB_USERNAME,
                "password": DB_PASSWORD,
                "driver": "org.postgresql.Driver",
            },
            transformation_ctx="load_data",
        )

        if dynamic_frame.count() == 0:
            logger.error("No data returned from the query")
            raise ValueError("Query returned empty result set")

        df = dynamic_frame.toDF()
        df = df.withColumn(
            "timestamp_unix",
            F.unix_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss").cast("long") * 1000,
        )
        logger.info(f"Loaded {df.count()} rows from PostgreSQL")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


# TRANSFORMATION LOGIC
def transform_parent_asin(df):
    """Transform data for parent_asin features."""
    try:
        window_latest = Window.partitionBy("parent_asin", "timestamp").orderBy(
            F.col("timestamp_unix").desc()
        )

        df_with_rn = (
            df.withColumn("rn", F.row_number().over(window_latest))
            .filter(F.col("rn") == 1)
            .drop("rn")
        )

        df_with_clean_price = df_with_rn.withColumn(
            "price_clean", F.col("price").cast(StringType())
        )

        aggregated_df = df_with_clean_price.groupBy(
            "parent_asin", "timestamp", "timestamp_unix"
        ).agg(
            F.first("main_category").alias("main_category"),
            F.flatten(F.collect_set(convert_categories_udf(F.col("categories")))).alias(
                "categories"
            ),
            F.first("price_clean").alias("price"),
            F.first("rating").alias("rating"),
        )

        window_fn = (
            lambda days: Window.partitionBy("parent_asin")
            .orderBy("timestamp_unix")
            .rangeBetween(-days * 86400000, -1)
        )

        result_df = aggregated_df.select(
            "parent_asin",
            F.col("timestamp").alias("event_timestamp"),
            F.col("main_category"),
            F.array_distinct(F.col("categories")).alias("categories"),
            F.col("price").cast(StringType()).alias("price"),
            F.count("*").over(window_fn(365)).alias("parent_asin_rating_cnt_365d"),
            F.avg("rating")
            .over(window_fn(365))
            .alias("parent_asin_rating_avg_prev_rating_365d"),
            F.count("*").over(window_fn(90)).alias("parent_asin_rating_cnt_90d"),
            F.avg("rating")
            .over(window_fn(90))
            .alias("parent_asin_rating_avg_prev_rating_90d"),
            F.count("*").over(window_fn(30)).alias("parent_asin_rating_cnt_30d"),
            F.avg("rating")
            .over(window_fn(30))
            .alias("parent_asin_rating_avg_prev_rating_30d"),
            F.count("*").over(window_fn(7)).alias("parent_asin_rating_cnt_7d"),
            F.avg("rating")
            .over(window_fn(7))
            .alias("parent_asin_rating_avg_prev_rating_7d"),
        ).distinct()

        logger.info(f"Transformed parent_asin data: {result_df.count()} rows")
        return result_df
    except Exception as e:
        logger.error(f"Failed to transform parent_asin data: {str(e)}")
        raise


def bucketize_seconds_diff(seconds):
    """Bucketize time differences."""
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


def calc_sequence_timestamp_bucket(timestamps, current_ts):
    """Calculate bucketed timestamps."""
    if not timestamps or current_ts is None:
        return [-1] * 10
    return [
        bucketize_seconds_diff(current_ts // 1000 - ts) if ts != -1 else -1
        for ts in timestamps
    ]


calc_bucket_udf = F.udf(calc_sequence_timestamp_bucket, ArrayType(IntegerType()))


def pad_timestamp_sequence(inp, sequence_length=10, padding_value=-1):
    """Pad timestamp sequence."""
    if inp is None or inp.strip() == "":
        return [padding_value] * sequence_length

    try:
        timestamps = [x.strip() for x in inp.split(",") if x.strip()]
        inp_list = []
        for ts in timestamps:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
                unix_ts = int(dt.timestamp())
                inp_list.append(unix_ts)
            except ValueError:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                    unix_ts = int(dt.timestamp())
                    inp_list.append(unix_ts)
                except ValueError:
                    continue
        padding_needed = sequence_length - len(inp_list)
        if padding_needed > 0:
            inp_list = [padding_value] * padding_needed + inp_list
        return inp_list[:sequence_length]
    except Exception:
        return [padding_value] * sequence_length


pad_timestamp_udf = F.udf(pad_timestamp_sequence, ArrayType(IntegerType()))


def transform_user(df):
    """Transform data for user features."""
    try:
        w_90d = (
            Window.partitionBy("user_id")
            .orderBy("timestamp_unix")
            .rangeBetween(-90 * 86400000, -1)
        )

        w_recent = (
            Window.partitionBy("user_id").orderBy("timestamp_unix").rowsBetween(-10, -1)
        )

        transformed_df = df.select(
            "user_id",
            F.col("timestamp").alias("event_timestamp"),
            F.col("timestamp_unix"),
            F.count("*").over(w_90d).alias("user_rating_cnt_90d"),
            F.avg("rating").over(w_90d).alias("user_rating_avg_prev_rating_90d"),
            F.concat_ws(",", F.collect_list("parent_asin").over(w_recent)).alias(
                "user_rating_list_10_recent_asin"
            ),
            F.concat_ws(
                ",",
                F.collect_list(
                    F.date_format(
                        F.to_utc_timestamp("timestamp", "UTC"),
                        "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
                    )
                ).over(w_recent),
            ).alias("user_rating_list_10_recent_asin_timestamp"),
        )

        transformed_df = transformed_df.withColumn(
            "item_sequence_ts",
            pad_timestamp_udf(F.col("user_rating_list_10_recent_asin_timestamp")),
        ).withColumn(
            "item_sequence_ts_bucket",
            calc_bucket_udf(F.col("item_sequence_ts"), F.col("timestamp_unix")),
        )

        window_dedup = Window.partitionBy("user_id", "event_timestamp").orderBy(
            F.size(F.split("user_rating_list_10_recent_asin", ",")).desc()
        )

        result_df = (
            transformed_df.withColumn("dedup_rn", F.row_number().over(window_dedup))
            .filter(F.col("dedup_rn") == 1)
            .drop("dedup_rn", "timestamp_unix")
        )

        logger.info(f"Transformed user data: {result_df.count()} rows")
        return result_df
    except Exception as e:
        logger.error(f"Failed to transform user data: {str(e)}")
        raise


# SPLIT TRAIN AND VAL
def split_train_val(df):
    """Split data into train and validation sets."""
    try:
        train_df = df.filter(F.col("source") == "train").drop("source")
        val_df = df.filter(F.col("source") == "val").drop("source")
        logger.info(f"Train set size: {train_df.count()}")
        logger.info(f"Validation set size: {val_df.count()}")

        # Validate train and validation sets
        train_min_max = train_df.select(
            F.min("timestamp").alias("min_timestamp"),
            F.max("timestamp").alias("max_timestamp"),
        ).collect()[0]

        val_min_max = val_df.select(
            F.min("timestamp").alias("min_timestamp"),
            F.max("timestamp").alias("max_timestamp"),
        ).collect()[0]

        logger.info(
            f"Train timestamp range: {train_min_max['min_timestamp']} to {train_min_max['max_timestamp']}"
        )
        logger.info(
            f"Val timestamp range: {val_min_max['min_timestamp']} to {val_min_max['max_timestamp']}"
        )

        train_items = train_df.select("parent_asin").distinct()
        val_items = val_df.select("parent_asin").distinct()
        items_not_in_train = val_items.join(train_items, "parent_asin", "left_anti")
        num_missing_items = items_not_in_train.count()

        train_users = train_df.select("user_id").distinct()
        val_users = val_df.select("user_id").distinct()
        users_not_in_train = val_users.join(train_users, "user_id", "left_anti")
        num_missing_users = users_not_in_train.count()

        if num_missing_items > 0:
            logger.error(
                f"Found {num_missing_items} items in validation set that do not appear in training set!"
            )
            raise ValueError(
                f"Found {num_missing_items} items in validation set that do not appear in training set!"
            )

        if num_missing_users > 0:
            logger.error(
                f"Found {num_missing_users} users in validation set that do not appear in training set!"
            )
            raise ValueError(
                f"Found {num_missing_users} users in validation set that do not appear in training set!"
            )

        return train_df, val_df
    except Exception as e:
        logger.error(f"Failed to split train and validation data: {str(e)}")
        raise


# CHECK AND DELETE S3 FOLDER
def check_and_delete_s3_folder(s3_client, bucket, prefix):
    """Check and delete S3 folder if it exists."""
    try:
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        logger.info(f"Checking if objects exist in s3://{bucket}/{prefix}")

        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" in response and len(response["Contents"]) > 0:
            logger.info(
                f"Objects found in s3://{bucket}/{prefix}. Deleting all objects..."
            )
            objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

            while response.get("IsTruncated", False):
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    ContinuationToken=response["NextContinuationToken"],
                )
                objects_to_delete.extend(
                    [{"Key": obj["Key"]} for obj in response.get("Contents", [])]
                )

            s3_client.delete_objects(
                Bucket=bucket, Delete={"Objects": objects_to_delete, "Quiet": True}
            )
            logger.info(
                f"Deleted {len(objects_to_delete)} objects from s3://{bucket}/{prefix}"
            )
        else:
            logger.info(f"No objects found in s3://{bucket}/{prefix}")
    except Exception as e:
        logger.error(f"Failed to check/delete S3 folder: {str(e)}")
        raise


# WRITE TO S3
def write_to_s3(df, prefix):
    """Write DataFrame to S3 in Parquet format."""
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        check_and_delete_s3_folder(s3_client, S3_BUCKET, prefix)

        dynamic_frame = DynamicFrame.fromDF(df, glueContext, "dynamic_frame")
        glueContext.write_dynamic_frame.from_options(
            frame=dynamic_frame,
            connection_type="s3",
            connection_options={
                "path": f"s3://{S3_BUCKET}/{prefix}/",
                "partitionKeys": [],
            },
            format="parquet",
            format_options={"compression": "snappy"},
            transformation_ctx=f"write_{prefix.replace('/', '_')}",
        )
        logger.info(f"Successfully wrote data to s3://{S3_BUCKET}/{prefix}/")
    except Exception as e:
        logger.error(f"Failed to write to S3: {str(e)}")
        raise


def read_ruleset_s3(path: str) -> str:
    s3 = boto3.client("s3")
    bucket, key = path.replace("s3://", "").split("/", 1)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def run_dqdl_check(df, ruleset_s3_path, glueContext, logger, name):
    dynf = DynamicFrame.fromDF(df, glueContext, f"dq_{name}")
    dqdl = read_ruleset_s3(ruleset_s3_path)
    dqf = EvaluateDataQuality.apply(
        frame=dynf,
        ruleset=dqdl,
        publishing_options={
            "dataQualityEvaluationContext": name,
            "enableDataQualityCloudWatchMetrics": False,
            "enableDataQualityResultsPublishing": False,
        },
    )
    results_df = dqf.toDF()
    fails = results_df.filter(F.col("Outcome") != "Passed")
    fails.show(truncate=False)

    for r in fails.collect():
        rule = r["Rule"] if "Rule" in r else str(r)
        outcome = r["Outcome"] if "Outcome" in r else "Unknown"
        fail_count = r["FailCount"] if "FailCount" in r else "N/A"
        eval_count = r["EvaluatedRows"] if "EvaluatedRows" in r else "N/A"
        logger.error(
            f"[DQDL] RULE FAIL ► {rule} | Outcome: {outcome} | FailCount: {fail_count} / EvaluatedRows: {eval_count}"
        )
    results_df.show(truncate=False)
    failed = results_df.filter(F.col("Outcome") != "Passed").count()
    if failed:
        logger.error(f"[DQDL][{name}] {failed} rule(s) failed")
        raise Exception(f"DQDL check failed for {name}: {failed} rule(s) did not pass")
    logger.info(f"[DQDL][{name}] All rules passed")


# MAIN EXECUTION
def main():
    try:
        # Load data
        df = load_data()

        # Check for duplicates
        duplicate_check = df.groupBy(
            "user_id",
            "parent_asin",
            "timestamp",
            "main_category",
            "categories",
            "price",
            "rating",
        ).count()
        duplicates = duplicate_check.filter(F.col("count") > 1)
        if duplicates.count() > 0:
            logger.warning("Found duplicate records in input data!")
            duplicates.show()
            df = df.dropDuplicates(
                [
                    "user_id",
                    "parent_asin",
                    "timestamp",
                    "main_category",
                    "categories",
                    "price",
                    "rating",
                ]
            )
            logger.info(f"After removing duplicates: {df.count()} rows")

        # Split train and validation data
        train_df, val_df = split_train_val(df)

        # Transform data
        parent_asin_df = transform_parent_asin(df)
        user_df = transform_user(df)

        run_dqdl_check(
            parent_asin_df,
            f"s3://{S3_BUCKET}/dq/parent_asin_stats.dqdl",
            glueContext,
            logger,
            "parent_asin_stats",
        )
        run_dqdl_check(
            user_df,
            f"s3://{S3_BUCKET}/dq/user_stats.dqdl",
            glueContext,
            logger,
            "user_stats",
        )

        # Check for duplicates in parent_asin_df
        duplicate_check = parent_asin_df.groupBy(
            "parent_asin", "event_timestamp"
        ).count()
        duplicates = duplicate_check.filter(F.col("count") > 1)
        if duplicates.count() > 0:
            logger.warning("Found duplicate records after parent_asin aggregation!")
            duplicates.show()

        # Write to S3 with exact paths from local code
        write_to_s3(train_df, "feature-store/train/train.parquet")
        write_to_s3(val_df, "feature-store/val/val.parquet")
        write_to_s3(parent_asin_df, "feature-store/parent_asin_rating_stats")
        write_to_s3(user_df, "feature-store/user_rating_stats")

        logger.info("Job completed successfully")
    except Exception as e:
        logger.error(f"Job failed: {str(e)}")
        raise
    finally:
        job.commit()


if __name__ == "__main__":
    main()
