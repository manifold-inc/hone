FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY miner/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY common /app/common

COPY miner /app/miner

ENV WALLET_NAME=default
ENV WALLET_HOTKEY=default
ENV WALLET_PATH=/root/.bittensor/wallets

EXPOSE 8091

CMD ["python", "-m", "miner.main"]