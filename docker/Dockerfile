# syntax=docker/dockerfile:1.6
#
# Hone training image. Bakes in CUDA 12.4 + cuDNN, Python 3.12, uv, and
# the hone package installed editable so source-tree edits are picked
# up on the next container restart without a rebuild.
#
# B200 / H200 require CUDA 12.4+; we pin 12.4.1 explicitly. torch>=2.7.1
# wheels for cu124 are pulled in by uv when the project is installed.
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    HF_HOME=/root/.cache/huggingface \
    HONE_DEFAULT_NCCL_TIMEOUT=600

# System deps. ``software-properties-common`` lets us add the deadsnakes
# PPA for Python 3.12; ``libgl1`` / ``libglib2.0-0`` are needed by some
# transitive deps (e.g. opencv-style libs that wandb pulls). git is
# required so that ``pip install`` can resolve git-based bittensor
# extras when the wheel index doesn't carry them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        tini \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# uv -- one binary, parallel resolves, much faster than pip for the
# bittensor + torch dep tree. Pinning to 0.5.x for reproducibility;
# ``--frozen`` would require a uv.lock which the project doesn't ship.
ADD --chmod=755 https://astral.sh/uv/install.sh /tmp/install-uv.sh
RUN UV_INSTALL_DIR=/usr/local/bin /tmp/install-uv.sh && rm /tmp/install-uv.sh

# Project source. We copy ``hone/`` (this directory's parent) into
# ``/app`` so the dockerfile can be built with ``-f hone/docker/Dockerfile``
# and ``--build-context=hone:./hone`` (or ``context: ..`` from the
# compose file's docker/ subdir).
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY src/hone/__init__.py /app/src/hone/__init__.py

# Resolve the dep set first so this layer caches across source edits.
# ``--no-install-project`` skips installing hone itself; that happens
# in the next layer once the full tree is copied. We use python 3.12.
RUN uv venv --python 3.12 /app/.venv \
    && . /app/.venv/bin/activate \
    && uv pip install --no-cache --upgrade pip wheel \
    && uv pip install --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        "torch>=2.7.1"

# Now bring in the rest of the source and install the project in
# editable mode. Editable install means rebuilds aren't needed for
# python edits -- a bind-mount of the host source over /app at runtime
# is also supported (see compose ``volumes:`` entry).
COPY . /app
RUN . /app/.venv/bin/activate \
    && uv pip install --no-cache -e .

# Make the venv the default Python everywhere downstream.
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# tini reaps zombies and forwards signals to the python process so
# SIGTERM from ``docker stop`` reaches the trainer cleanly (the trainer
# already has a SIGTERM handler that does graceful shutdown -- see
# ``base_node._setup_signal_handlers``).
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
