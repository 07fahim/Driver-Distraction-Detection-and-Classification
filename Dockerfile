FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

# Create folders and set permissions BEFORE switching user
RUN mkdir -p temp_uploads temp_results && \
    chown -R 1000:1000 temp_uploads temp_results

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]