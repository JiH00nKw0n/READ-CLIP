FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        wget \
        git \
        unzip \
        vim \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and activate Python 3.10 venv
RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install PyTorch with CUDA 12.1 support
RUN pip install --upgrade pip
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY . /app/

# Make setup.sh executable (if exists)
RUN chmod +x setup.sh || true

# Install requirements
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Create required directories
RUN mkdir -p /app/output/READ-CLIP /app/logs /app/data

# Environment variables
ENV LOG_DIR=/app/logs
ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app