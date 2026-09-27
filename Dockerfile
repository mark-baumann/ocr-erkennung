# GPU-Image für DeepSeek-OCR Studio (Streamlit-GUI + headless CLI).
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
RUN pip install -e ".[app]" -r requirements-gpu.txt

COPY . .

# Port (pro App anpassen: 8513). ARG allein reicht nicht: CMD/HEALTHCHECK
# laufen zur Container-Laufzeit und lesen $PORT vom Environment — als ENV
# re-exportieren (Muster wie im rag-agent-langgraph Dockerfile).
ARG PORT=8513
ENV PORT=$PORT
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ['PORT'])"

# Streamlit-GUI als Default; headless CLI weiterhin: docker run ... python -m dsocr.cli ...
CMD ["sh", "-c", "streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]