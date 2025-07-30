FROM ghcr.io/mlflow/mlflow:v2.16.2

RUN apt-get -y update && \
    apt-get -y install python3-dev build-essential pkg-config && \
    pip install --upgrade pip && \
    pip install psycopg2-binary boto3

CMD ["bash"]