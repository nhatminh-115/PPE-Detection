FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện lõi cho hệ điều hành (chủ yếu phục vụ OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Chỉ mở cổng 8000 cho FastAPI, khai tử cổng 8501 của Streamlit
EXPOSE 8000

# Trỏ thẳng vào file core API
CMD ["python", "api.py"]