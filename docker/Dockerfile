# Change the CUDA version according to your cluster's driver version
# Check with: nvidia-smi | grep "CUDA Version"
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    # Required for pdf2image and pytesseract
    poppler-utils \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3.10-dev python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools pip

# Point python to 3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

RUN python3 --version && pip --version

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy project source
COPY . /app

# Default command: run training
CMD ["python", "-m", "src.model.train"]

# ──────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -f docker/Dockerfile -t ocr-corrector:v1 .
#
# Convert to Singularity .sif (run on cluster or machine with Singularity):
#   singularity build ocr-corrector.sif docker-daemon://ocr-corrector:v1
#
# Submit training job:
#   sbatch train.srm
# ──────────────────────────────────────────────────────────────────────────────
