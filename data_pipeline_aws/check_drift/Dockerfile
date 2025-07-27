FROM python:3.12-slim

# Install system dependencies (for psycopg2, pandas, etc.)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    boto3==1.34.69 \
    pandas==2.2.2 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.10 \
    evidently==0.6.7 \
    pyarrow==14.0.2

# Set working directory
WORKDIR /app

# Copy Python script into container
COPY app.py /app/app.py

# Run the script
CMD ["python", "app.py"]