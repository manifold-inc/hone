FROM ubuntu:24.04 AS base

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    iptables \
    iproute2 \
    cgroup-tools \
    libcap2-bin \
    apparmor \
    apparmor-utils \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://gvisor.dev/archive.key | gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | \
    tee /etc/apt/sources.list.d/gvisor.list > /dev/null && \
    apt-get update && \
    apt-get install -y runsc || true && \
    rm -rf /var/lib/apt/lists/*

FROM base AS python-deps

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

FROM python-deps AS app

WORKDIR /app

COPY . .

RUN mkdir -p /var/log/sandbox-runner && \
    mkdir -p /tmp/sandbox-jobs && \
    mkdir -p /etc/sandbox-runner

RUN chmod +x setup.sh || true

EXPOSE 8443
EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8443/health')" || exit 1

CMD ["python3", "main.py", "--config", "/etc/sandbox-runner/config.yaml"]