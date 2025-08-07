
# Stage 1: Builder (for installing dependencies)
FROM python:3.11-slim as builder

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage 2: Production (minimal runtime)
FROM python:3.11-slim

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:$PATH" \
    PORT=8000

# Install only curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create the same user in production image
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy installed packages from builder stage
COPY --from=builder /home/app/.local /home/app/.local

# Copy your application files
COPY --chown=app:app main.py .
COPY --chown=app:app requirements.txt .

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose the port your app runs on
EXPOSE $PORT

# Start your application
CMD ["python", "main.py"]
