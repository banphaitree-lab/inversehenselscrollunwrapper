FROM python:3.11-slim

LABEL maintainer="LoTT Framework"
LABEL description="Vesuvius Challenge - Inverse Hensel Scroll Unwrapper"
LABEL version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-image>=0.21.0 \
    pillow>=9.0.0 \
    vesuvius>=0.1.0

WORKDIR /app

# Copy source files
COPY inverse_hensel_unwrapper.py .
COPY run.sh .
RUN chmod +x run.sh

# Create output directory
RUN mkdir -p /app/outputs

# Entry point
ENTRYPOINT ["./run.sh", "/app/outputs"]
