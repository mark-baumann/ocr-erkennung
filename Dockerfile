# GPU-Image für die DeepSeek-OCR-Batch-Pipeline.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip fonts-dejavu-core libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Torch zuerst, damit der Layer beim Ändern des App-Codes im Cache bleibt.
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 torchvision==0.21.0 \
       --index-url https://download.pytorch.org/whl/cu118

COPY pyproject.toml requirements-gpu.txt README.md ./
COPY dsocr ./dsocr
RUN pip install -e . -r requirements-gpu.txt

COPY . .

ENTRYPOINT ["python", "-m", "dsocr.cli"]
