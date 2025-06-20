Trước khi tạo pipeline ta cần tạo rds postgress trên aws với engine version = postgresql 17.4-R1, template: free tier
với DB instance identifier: simulate-oltp-db, master username: postgres, password: postgres, public access: yes

![systempipline](images/data_pipeline/rds.png)

sau đó ta kết nối đến rds đã tạo bằng dbeaver để  tạo db name
CREATE DATABASE raw_data;
CREATE DATABASE registry_feature_store;
với raw_data là db giả lập oltp, dự liệu được upload từ notebooks/02-upload-data.ipynb
registry_feature_store là registry lưu trữ của feature store
![systempipline](images/data_pipeline/dbeaver.png)

Tiếp theo ta upload dữ liệu vào db raw_data, trước tiên chạy notebooks/01-prep-data.ipynb để sampling data từ dữ liệu raw, giữ lại user và item có tương tác tối thiểu, sau đó run notebooks/02-upload-data.ipynb để upload dữ liệu đã sampling lên db_rawdata, đồng thời khi run, nó sẽ upload holdout data lên S3, đây là dữ liệu giả lập để test luồng realtime cdc
![systempipline](images/data_pipeline/data_in_rds.png)

upload file dqdl glue data quality lên s3 để glue có thể pull về để check data quality trước khi upload data vào offline store:
aws s3 cp data_pipeline_aws/glue_data_quality/parent_asin_stats.dqdl s3://recsys-ops/dq/parent_asin_stats.dqdl
aws s3 cp data_pipeline_aws/glue_data_quality/user_stats.dqdl s3://recsys-ops/dq/user_stats.dqdl

Dùng glue 3.0 với type: spark, cấu hình các jop parameter như sau: 
Key
--AWS_REGION

Value - optional
ap-southeast-1

Key
--DB_CONNECTION

Value - optional
jdbc:postgresql://simulate-oltp-db.cdkwg6wyo7r8.ap-southeast-1.rds.amazonaws.com:5432/raw_data

Key
--DB_PASSWORD

Value - optional
postgres

Key
--DB_USERNAME

Value - optional
postgres

Key
--JOB_NAME

Value - optional
FeastFeatureTransformJob

Key
--S3_BUCKET

Value - optional
recsys-ops

Key
--TABLE_NAME

Value - optional
public.reviews

-----------

