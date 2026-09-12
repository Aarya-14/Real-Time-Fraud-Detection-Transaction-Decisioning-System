
# --- Stage 1: Builder (Install Dependencies) ---
# Use a full Python image for building, as it contains necessary tools (like gcc)
FROM python:3.11 as builder

WORKDIR /build

# Install essential system build dependencies required for complex packages (like psycopg2 or numpy/scikit-learn)
# libgomp1 is crucial for multi-threaded math libraries (like LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies. We use /root/.local/ to avoid
# problems with virtual environments and permission.
COPY requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# --- Stage 2: Runtime (Minimal & Secure) ---
# Use the slim base image to reduce the final container size significantly
FROM python:3.11-slim

# Install only the runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 frauduser && \
    mkdir -p /app/models /app/logs && \
    chown -R frauduser:frauduser /app

WORKDIR /app

# Copy Python packages from builder's home directory to the new non-root user's home directory
COPY --from=builder /root/.local /home/frauduser/.local

# Copy application code and set ownership to the non-root user
COPY --chown=frauduser:frauduser src/ ./src/
COPY --chown=frauduser:frauduser models/ ./models/
COPY --chown=frauduser:frauduser config/ ./config/

# Set environment variables for non-root user path and Python behavior
ENV PATH=/home/frauduser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Switch to non-root user
USER frauduser

# Expose Cloud Run port (optional but good practice)
EXPOSE 8080

# Start application using the Exec form (CMD ["command", "arg1", "arg2"]) for proper signal handling.
# --workers 1 is critical for Cloud Run as it handles concurrency itself.
CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info"]
