# Multi-stage production Dockerfile for InsureAI Backend
# Pin exact Python minor version to match training environment (Python 3.10.x)
FROM python:3.10.15-slim AS backend

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --system insureai && useradd --system --gid insureai insureai

# Install Python dependencies (exact pins from requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and required artifacts
COPY backend/ ./backend/
COPY models/ ./models/
COPY schemas/ ./schemas/
COPY data/processed/ ./data/processed/
# data/real/ needed for analytics, monitoring, and fairness endpoints
COPY data/real/ ./data/real/

# Switch to non-root user before starting the application
USER insureai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
