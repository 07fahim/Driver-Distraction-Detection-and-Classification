FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, numpy, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a minimal onnxruntime wrapper if needed
RUN python3 -c "import onnxruntime; print('ONNX Runtime loaded successfully')" || echo "ONNX Runtime import failed"

# Copy application code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]