FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including FFmpeg for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    x264 \
    libx264-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify FFmpeg installation
RUN ffmpeg -version && \
    ffmpeg -encoders 2>&1 | grep h264 || echo "Warning: H.264 encoder may not be available"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify ONNX Runtime installation
RUN python3 -c "import onnxruntime; print('✅ ONNX Runtime loaded successfully')"
RUN python3 -c "import cv2; print('✅ OpenCV loaded successfully')"

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "--worker-class", "sync", "app:app"]