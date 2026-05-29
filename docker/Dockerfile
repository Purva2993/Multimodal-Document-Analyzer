FROM python:3.11-slim

  # tesseract-ocr is the system binary Whisper needs — not the Python package
  RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      libgl1 \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

  WORKDIR /app

  COPY requirements.txt .

  # CPU-only torch first — avoids pulling 1GB+ of NVIDIA CUDA packages
  RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

  RUN pip install --no-cache-dir -r requirements.txt

  COPY . .
  
  EXPOSE 8000 7860
