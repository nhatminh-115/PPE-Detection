FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements.txt

COPY . .

# Pre-bake Real-ESRGAN weights so the first crop upload does not trigger a
# network download at runtime.
RUN mkdir -p model_cache && \
    if [ ! -s model_cache/RealESRGAN_x4plus.pth ]; then \
      wget -q -O model_cache/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth; \
    fi

EXPOSE 8000

CMD ["python", "main.py"]