# L4 GPU + CUDA 12.1 + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV BNB_CUDA_VERSION=121

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl wget \
    build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 핵심 패키지 먼저 설치 (버전 호환성)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# 호환되는 버전으로 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# StoryTeller-AI 소스 코드 복사
COPY . /app

# 포트 노출
EXPOSE 8000 8001

CMD ["python3", "start_server.py"]