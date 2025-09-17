FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/

COPY validator/ ./validator/

RUN mkdir -p /data

ENV PYTHONPATH=/app

CMD ["python", "validator/main.py"]