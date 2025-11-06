FROM python:3.11-slim

WORKDIR /app

# Install build tools + FFmpeg + OpenCV deps
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
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
    libavutil-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build OpenCV 4.10.0 from source with FFmpeg
RUN git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git && \
    mkdir -p opencv/build && cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_apps=OFF \
          -D BUILD_opencv_python3=ON \
          -D BUILD_opencv_python2=OFF \
          -D PYTHON3_EXECUTABLE=$(which python) \
          -D WITH_FFMPEG=ON \
          -D WITH_GSTREAMER=OFF \
          -D WITH_V4L=OFF \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd ../../ && rm -rf opencv

# Install Python packages (NO opencv-python)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]