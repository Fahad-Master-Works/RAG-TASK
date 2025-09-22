
# ---- Base image
FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements first (leverage docker layer cache)
COPY requirements.txt ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and data
COPY app ./app
COPY data ./data

# Create storage dir for FAISS index
RUN mkdir -p /app/storage
VOLUME ["/app/storage", "/app/data"]

# Env defaults
ENV PDF_PATH=/app/data/AI.pdf \
    INDEX_DIR=/app/storage \
    EMBEDDING_MODEL=intfloat/multilingual-e5-base \
    GROQ_MODEL=llama-3.1-8b-instant \
    TOP_K=4 \
    CHUNK_SIZE=900 \
    CHUNK_OVERLAP=120 \
    MAX_CONTEXT_CHARS=12000 \
    REBUILD_INDEX=0

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
