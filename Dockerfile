FROM python:3.11-slim

WORKDIR /app

# Force OpenCV to use system FFmpeg
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist|file,rtp,udp,tcp,http"
ENV OPENCV_FFMPEG_WRITER_OPTIONS="preset|veryfast"

# Install system deps + ffmpeg
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]