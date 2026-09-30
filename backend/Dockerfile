FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY backend /app/backend
RUN pip install --upgrade pip && pip install .

COPY datasets /app/datasets

EXPOSE 8000
CMD ["uvicorn", "pe.axiz.payment_knowledge.main:app", "--host", "0.0.0.0", "--port", "8000"]
