FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir -r <(grep -v 'faiss-cpu' pyproject.toml)
RUN pip install --no-cache-dir faiss-cpu

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads outputs data

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
