FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    libgomp1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
ENV PYARROW_IGNORE_TIMEZONE=1

RUN pip install --no-cache-dir \
    grpcio==1.62.1 \
    grpcio-tools==1.62.1 \
    grpcio-health-checking==1.62.1 \
    grpcio-reflection==1.62.1 \
    protobuf==4.25.3 \
    pyarrow==10.0.1 \
    pandas==1.5.3 \
    pyspark==3.5.0 \
    boto3==1.37.1 \
    s3fs==2024.6.1 \
    sqlalchemy==1.4.54 \
    psycopg2-binary==2.9.10 \
    feast[aws]==0.47.0

RUN apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app