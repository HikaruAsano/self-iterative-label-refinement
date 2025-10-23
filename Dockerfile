FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG APP_DIR="/app"
ARG WANDB_API_KEY
ARG WANDB_ENTITY
ARG HUGGINGFACE_HUB_TOKEN
ARG E2B_API_KEY

# Add arguments for user and group IDs with default values
ARG USER_ID=1009
ARG GROUP_ID=1009

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEB_PYTHON_INSTALL_LAYOUT=deb \
    HOME=/home/appuser \
    PYTHONPATH=${APP_DIR}/src:$PYTHONPATH \
    CACHE_DIR=/home/appuser/.cache \
    WANDB_CACHE_DIR=$CACHE_DIR/wandb \
    WANDB_DATA_DIR=$CACHE_DIR/data \
    HF_HOME=$CACHE_DIR/transformer \
    HF_DATASETS_CACHE=$CACHE_DIR/datasets \
    HYDRA_FULL_ERROR=1 

# Install Node.js 18.x from NodeSource repository and other dependencies
RUN apt update \
    && apt install -y --no-install-recommends \
       curl \
       wget \
       git \
       pkg-config \
       libssl-dev \
       cmake \
       build-essential \
       software-properties-common \
       ca-certificates \
    && apt update \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group with specified UID and GID
RUN groupadd -g ${GROUP_ID} appgroup \
 && useradd -u ${USER_ID} -g appgroup -m appuser \
 && chown -R appuser:appgroup /home/appuser \
 && mkdir -p /app \
 && chown -R appuser:appgroup /app \
 && mkdir -p /home/appuser/.cache \
 && chmod -R 777 /home/appuser/.cache \
 && mkdir -p /home/appuser/.local/share/uv \
 && chown -R appuser:appgroup /home/appuser/.local \
 && chown -R appuser:appgroup /home/appuser/.cache

# Set the working directory
WORKDIR ${APP_DIR}

# Switch to the non-root user
USER appuser

# Rust & uv install for appuser
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/appuser/.cargo/bin:/home/appuser/.local/bin:$PATH"

COPY --chown=appuser:appgroup . .

# Setup virtual environment and install dependencies in a single RUN command
RUN uv venv .venv && \
    uv sync

ENV PATH="/app/.venv/bin:$PATH"

CMD ["/bin/bash"]