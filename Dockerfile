FROM python:3.11-slim

# Install system dependencies including libgomp
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ocr_service.py .

# Expose default port (Railway still injects PORT dynamically)
EXPOSE 8080

# Run with gunicorn (bind to Railway PORT if provided)
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-8080} --timeout 300 ocr_service:app"]
