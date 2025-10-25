FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    psycopg2-binary==2.9.9 \
    pandas==2.2.2 \
    plotly==5.24.1 \
    substrate-interface==1.7.9 \
    websocket-client==1.8.0 \
    xxhash==3.5.0

COPY app.py .

EXPOSE 8501

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py"]
