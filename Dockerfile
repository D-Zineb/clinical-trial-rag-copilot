# Multi-stage production container for Cardiology GraphRAG API & UI
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Install dependencies into virtual environment
RUN uv sync --frozen --no-cache

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy virtual environment and project code
COPY --from=builder /app/.venv /app/.venv
COPY . /app

EXPOSE 8000 8501

# Default: Launch FastAPI REST API Service
CMD ["python", "-m", "uvicorn", "clinical_trial_rag_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
