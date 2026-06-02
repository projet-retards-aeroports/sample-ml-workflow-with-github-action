# 1. Base Image
FROM python:3.11-slim

WORKDIR /app

# 2. System Dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy Manifest
COPY requirements.txt .

# 4. INSTALLATION (With Debugging)
# We 'cat' the file to the build logs so you can SEE if pandas is listed.
RUN echo "===== CHECKING REQUIREMENTS =====" && \
    cat requirements.txt && \
    echo "=================================" && \
    python -m pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt && \

# 5. Copy Application Code
COPY . .

# 6. Runtime Config
ENV PYTHONPATH=/app
CMD ["python", "app/train.py"]
