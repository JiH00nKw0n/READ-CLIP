FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    unzip \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 파일 복사
COPY . /app/

# setup.sh 실행 권한 부여
RUN chmod +x setup.sh

# 요구 사항 설치
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# 기본 디렉토리 생성
RUN mkdir -p /app/output/READ-CLIP /app/logs /app/data

# 필요시 환경 변수 설정
ENV LOG_DIR=/app/logs
ENV PYTHONPATH=/app:$PYTHONPATH

# 작업 디렉토리 설정
WORKDIR /app

# 기본 진입점 설정
CMD ["bash"] 