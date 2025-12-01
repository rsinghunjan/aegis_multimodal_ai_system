# Minimal image for running Aegis components (safety, federated demo, metrics)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# System deps for common scientific packages and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY aegis_multimodal_ai_system/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copy the repository code
COPY . /app

# Expose prometheus metrics port (default)
EXPOSE 8000

# Default command: keep container alive; override with federated server/client or other entrypoints.
CMD ["tail", "-f", "/dev/null"]
