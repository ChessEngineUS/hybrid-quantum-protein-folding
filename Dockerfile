# Hybrid Quantum-AI Protein Folding - Docker Image
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY . .
RUN pip install --no-cache-dir --user -e .

# Runtime stage
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Hybrid Quantum-AI Protein Folding"
LABEL org.opencontainers.image.description="Hybrid quantum-classical framework for protein structure prediction"
LABEL org.opencontainers.image.authors="Tommaso R. Marena"
LABEL org.opencontainers.image.source="https://github.com/ChessEngineUS/hybrid-quantum-protein-folding"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 hqpf && \
    mkdir -p /app /data /output && \
    chown -R hqpf:hqpf /app /data /output

# Copy installed packages from builder
COPY --from=builder --chown=hqpf:hqpf /root/.local /home/hqpf/.local

# Copy application code
WORKDIR /app
COPY --chown=hqpf:hqpf . .

# Switch to non-root user
USER hqpf

# Add local bin to PATH
ENV PATH=/home/hqpf/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Set environment variables
ENV HQPF_DATA_DIR=/data
ENV HQPF_OUTPUT_DIR=/output
ENV USE_QUANTUM_SIMULATOR=true

# Expose port for future web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hqpf; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "hqpf.cli"]
