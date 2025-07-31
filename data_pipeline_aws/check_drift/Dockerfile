FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    boto3==1.34.69 \
    pandas==2.2.2 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.10 \
    evidently==0.6.7 \
    pyarrow==14.0.2

WORKDIR /app

COPY app.py /app/app.py

CMD ["python", "app.py"]