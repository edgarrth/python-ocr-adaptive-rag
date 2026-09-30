FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY backend /app/backend

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install . \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /root/.cache/pip

COPY datasets /app/datasets

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "pe.axiz.payment_knowledge.main:app", "--host", "0.0.0.0", "--port", "8000"]
