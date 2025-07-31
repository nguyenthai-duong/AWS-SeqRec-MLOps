FROM rayproject/ray:2.44.1-py311

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
