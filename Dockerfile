FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if required by native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ /app/src/
# Note: The model directory must exist before building, which means train.py must be run first!
COPY models/ /app/models/

# EXPOSE port
EXPOSE 8080

# Environment variables
ENV MODEL_DIR=/app/models/brain-tumor-slm
ENV PYTHONUNBUFFERED=1

# Start FastAPI via uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
