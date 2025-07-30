FROM python:3.11.9-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
ENV PYARROW_IGNORE_TIMEZONE=1

ENV MAVEN_OPTS=-Dmaven.repo.local=/root/.m2/repository
ENV IVY_HOME=/root/.ivy2

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \ 
    openjdk-17-jdk-headless \
    libgomp1 \
    gcc \
    maven \
    ant \        
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.m2/repository && mkdir -p /root/.ivy2/cache

RUN mkdir -p /opt/ivy && \
    curl -fSL https://archive.apache.org/dist/ant/ivy/2.5.1/apache-ivy-2.5.1-bin.tar.gz \
    | tar -xz -C /opt/ivy --strip-components=1 && \
    ln -s /opt/ivy/bin/ivy /usr/local/bin/ivy
    
WORKDIR /app

COPY uv.lock pyproject.toml ./

RUN uv sync --group features --group pipeline

# Bật venv cho mọi lệnh Python/Papermill
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages"

RUN /app/.venv/bin/python -m ipykernel install --name python3 \
      --display-name "Python 3 (venv)" --prefix=/usr/local
COPY src/ ./src/
COPY feature_store/feature_store.yaml ./feature_store/feature_store.yaml

WORKDIR /app/src/feature_engineer
RUN mkdir -p papermill-output

