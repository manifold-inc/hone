# syntax=docker/dockerfile:1.6
#
# Hone training image. Bakes in CUDA 12.6 + cuDNN, Python 3.12, uv, and
# the hone package installed editable so source-tree edits are picked
# up on the next container restart without a rebuild.
#
# B200 / H200 require CUDA 12.4+; we pin 12.6.3 explicitly because
# pytorch.org's cu124 wheel index caps at torch 2.6.0 while
# pyproject.toml requires torch>=2.7.1 -- the cu126 index has 2.7.x
# wheels and remains B200-compatible (12.6 supersedes 12.4 ABI).
#
# Use the *runtime* (not -devel) CUDA variant: the PyTorch wheel
# bundles every CUDA shared lib it needs and Triton bundles its own
# ``ptxas`` for JIT, so we don't need ``nvcc`` / headers / static
# libs. That cuts image size by ~3 GiB and shaves ~30-60s off the
# layer-export step that finalises the build.
ARG CUDA_VERSION=12.6.3
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

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
# and ``context: ..`` from the compose file's docker/ subdir. The
# build context is filtered by ``hone/.dockerignore`` so dataset
# shards / virtualenvs / logs don't get streamed to the daemon.
#
# IMPORTANT: the venv lives at /opt/hone-venv, NOT under /app. The
# runtime compose mounts the host hone/ at /app:rw so that source
# edits land instantly; if the venv were at /app/.venv it would be
# shadowed by whatever (or nothing) the host has at hone/.venv.
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY src/hone/__init__.py /app/src/hone/__init__.py

# Resolve the dep set first so this layer caches across source edits.
# Torch is pinned to the cu126 wheel index so we get the matching
# CUDA runtime baked into the image (cu124 caps at torch 2.6.0).
# ``--index-strategy unsafe-best-match`` lets uv consider PyPI for
# everything that's NOT torch (transitive deps that aren't on the
# pytorch index), otherwise the cu126 index would shadow PyPI for
# unrelated packages.
RUN uv venv --python 3.12 /opt/hone-venv \
    && . /opt/hone-venv/bin/activate \
    && uv pip install --no-cache --upgrade pip wheel \
    && uv pip install --no-cache \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch>=2.7.1"

# Bring in the rest of the source and install the project in
# editable mode. Editable means future ``COPY``-bound or bind-mounted
# python edits are picked up live -- no rebuild needed.
COPY . /app
RUN . /opt/hone-venv/bin/activate \
    && uv pip install --no-cache -e .

# Make the venv the default Python everywhere downstream. Putting
# /opt/hone-venv first on PATH means ``python``, ``torchrun``, etc.
# resolve to the container's interpreter even when /app is bind-
# mounted from the host.
ENV PATH="/opt/hone-venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/hone-venv"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# tini reaps zombies and forwards signals to the python process so
# SIGTERM from ``docker stop`` reaches the trainer cleanly (the trainer
# already has a SIGTERM handler that does graceful shutdown -- see
# ``base_node._setup_signal_handlers``).
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
