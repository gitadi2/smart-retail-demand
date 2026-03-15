FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY models/ models/
COPY data/ data/

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "src.api.forecasting_api:app", "--host", "0.0.0.0", "--port", "8000"]
