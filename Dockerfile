FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including execstack to fix ONNX Runtime
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    execstack \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix ONNX Runtime executable stack issue
# This clears the executable stack flag from all shared objects
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime -name "*.so" -type f -exec execstack -c {} \; 2>/dev/null || true

# Copy application code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]