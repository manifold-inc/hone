FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
ca-certificates \
curl \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN useradd -m -u 1000 orchestrator && mkdir -p /orchestrator && chown -R orchestrator:orchestrator /orchestrator
USER orchestrator

ENV ORCH_DATA_ROOT=/orchestrator/data \
ORCH_OUT_ROOT=/orchestrator/out \
ORCH_TMP_ROOT=/orchestrator/tmp

EXPOSE 8080
CMD ["python", "-m", "app.main"]