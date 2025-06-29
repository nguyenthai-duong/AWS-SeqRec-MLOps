# Real-Time Data Pipeline Setup on AWS

This guide provides detailed instructions for setting up a data pipeline on AWS using RDS PostgreSQL, AWS Glue, Kinesis, DMS Serverless, and Lambda to support both batch processing and real-time data ingestion via Change Data Capture (CDC). The pipeline simulates an Online Transaction Processing (OLTP) system, processes the data, and stores engineered features in a feature store for downstream applications.

## Prerequisites

- An AWS account with administrative privileges.
- Basic familiarity with AWS services (RDS, Glue, Kinesis, DMS, S3).
- Tools installed: DBeaver (for database management), AWS CLI (configured with appropriate credentials).
- A local clone of the repository containing the necessary notebooks and scripts under `notebooks/` and `data_pipeline_aws/`.

## Step 1: Set Up RDS PostgreSQL

Create an RDS PostgreSQL instance with the following configuration:

- **Engine**: PostgreSQL 17.4-R1
- **Template**: Free Tier
- **DB Instance Identifier**: `simulate-oltp-db`
- **Master Username**: `postgres`
- **Master Password**: `postgres`
- **Public Access**: Enabled (set to `Yes`)

![RDS Setup](../images/data_pipeline/rds.png)

Once the RDS instance is created, note the endpoint (e.g., `simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com`).

## Step 2: Create Databases

Connect to the RDS instance using DBeaver with the credentials provided above (`postgres`/`postgres`). Run the following SQL commands to create two databases:

```sql
CREATE DATABASE raw_data;
CREATE DATABASE registry_feature_store;
```

- **`raw_data`**: Simulates an OLTP database for storing raw transactional data.
- **`registry_feature_store`**: Stores the registry for the feature store.

![DBeaver Database Creation](../images/data_pipeline/dbeaver.png)

## Step 3: Upload Data to `raw_data`

1. **Prepare Data**:
   - Run the notebook `notebooks/01-prep-data.ipynb` to sample data from the raw dataset. This step filters users and items with a minimum number of interactions to create a clean dataset.
   
2. **Upload Data**:
   - Run the notebook `notebooks/02-upload-data.ipynb` to upload the sampled data to the `raw_data` database, specifically to the `public.reviews` table.
   - This notebook also uploads holdout data to an S3 bucket (`s3://recsys-ops/`) for testing the real-time CDC pipeline.

![Data in RDS](../images/data_pipeline/data_in_rds.png)

## Step 4: Upload Glue Data Quality Scripts to S3

Upload the AWS Glue Data Quality Definition Language (DQDL) files to S3 for data quality checks before loading data into the offline feature store:

```bash
aws s3 cp data_pipeline_aws/glue_data_quality/parent_asin_stats.dqdl s3://recsys-ops/dq/parent_asin_stats.dqdl
aws s3 cp data_pipeline_aws/glue_data_quality/user_stats.dqdl s3://recsys-ops/dq/user_stats.dqdl
```

These scripts will be used by AWS Glue to validate data quality.

## Step 5: Configure AWS Glue Job

Create an AWS Glue job (version 3.0, type: Spark) with the following job parameters:

| **Key**            | **Value**                                                                 |
|--------------------|---------------------------------------------------------------------------|
| `--AWS_REGION`     | `ap-southeast-1`                                                          |
| `--DB_CONNECTION`  | `jdbc:postgresql://simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/raw_data` |
| `--DB_PASSWORD`    | `postgres`                                                                |
| `--DB_USERNAME`    | `postgres`                                                                |
| `--JOB_NAME`       | `FeastFeatureTransformJob`                                                |
| `--S3_BUCKET`      | `recsys-ops`                                                              |
| `--TABLE_NAME`     | `public.reviews`                                                          |

This Glue job processes data from the `raw_data` database and transforms it into features for the feature store. The corresponding code resides in `data_pipeline_aws/glue.py`.

## Step 6: Configure Streaming Data for CDC

### 6.1 Update RDS Parameter Group

To enable logical replication for real-time CDC, modify the RDS instance’s parameter group:

1. In the RDS console, go to **Modify DB Instance** and select the parameter group.
2. Update the parameter group with the following settings:

```
rds.logical_replication = 1
shared_preload_libraries = pg_stat_statements,pg_tle,pglogical
```

3. Apply the updated parameter group to the RDS instance.
4. Reboot the RDS instance to apply the changes.

![Parameter Group Configuration](../images/data_pipeline/parameter_group.png)

### 6.2 Enable `pglogical` Extension

Connect to the RDS instance using DBeaver or a similar tool and run the following SQL command to enable the `pglogical` extension:

```sql
CREATE EXTENSION IF NOT EXISTS pglogical;
```

### 6.3 Set Up Kinesis Data Stream

1. Create a Kinesis Data Stream in the AWS console to handle streaming data.
![Kinesis Data Stream](../images/data_pipeline/kinesis.png)
2. Create a VPC endpoint for Kinesis to allow secure communication within the VPC.
3. Add an HTTPS security group to the VPC endpoint for Kinesis to ensure secure connectivity.

### 6.4 Configure AWS DMS Serverless

#### 6.4.1 Create DMS Role

Create an IAM role for AWS DMS with the necessary permissions for accessing the RDS instance and Kinesis stream.

![DMS VPC Role](../images/data_pipeline/dms_vpc_role.png)

#### 6.4.2 Create Source Endpoint

Set up a source endpoint in AWS DMS for the RDS PostgreSQL database:

- **Endpoint Type**: Source
- **Database**: `raw_data`
- **Connection Details**: Use the RDS endpoint, username (`postgres`), and password (`postgres`).
- Test the connection to ensure it is successful.

![Source Endpoint 1](../images/data_pipeline/src_endpoint1.png)
![Source Endpoint 2](../images/data_pipeline/src_endpoint2.png)
![Source Endpoint 3](../images/data_pipeline/src_endpoint3.png)

#### 6.4.3 Create Target Endpoint

Set up a target endpoint in AWS DMS for the Kinesis Data Stream:

- **Endpoint Type**: Target
- **Service**: Kinesis
- Test the connection to ensure it is successful.

![Target Endpoint 1](../images/data_pipeline/des_endpoint1.png)
![Target Endpoint 2](../images/data_pipeline/des_endpoint2.png)

## Step 7: 




# Setup lambda
cd data_pipeline_aws/lambda

docker build -t feast-lambda:v1 .

export $(grep -v '^#' ../../.env | xargs)

aws ecr get-login-password --region ap-southeast-1 | \
docker login --username AWS --password-stdin 796973475591.dkr.ecr.ap-southeast-1.amazonaws.com

docker tag feast-lambda:v1 796973475591.dkr.ecr.ap-southeast-1.amazonaws.com/datn/feast-lambda:v1
docker push 796973475591.dkr.ecr.ap-southeast-1.amazonaws.com/datn/feast-lambda:v1