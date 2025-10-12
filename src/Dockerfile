# L4 GPU + CUDA 12.1 + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV BNB_CUDA_VERSION=121

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl wget \
    build-essential libssl-dev \
    espeak-ng espeak-ng-data \
    ffmpeg \
    file \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

# Zonos 설치 (수정된 방법)
RUN git clone https://github.com/Zyphra/Zonos.git /app/Zonos && \
    cd /app/Zonos && \
    pip install --no-cache-dir -e . && \
    echo "Zonos 설치 완료" && \
    python3 -c "from zonos.model import Zonos; print('Zonos import 성공')"

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000 8001

# 디렉토리 이동
WORKDIR /app/src

CMD ["python3", "start_server.py"]