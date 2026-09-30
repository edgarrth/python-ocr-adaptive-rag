FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY raglight_service/pyproject.toml /app/pyproject.toml
COPY raglight_service/src /app/src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install . \
    && rm -rf /root/.cache/pip

EXPOSE 8010
CMD ["python", "-m", "uvicorn", "pe.axiz.raglight_service.main:app", "--host", "0.0.0.0", "--port", "8010"]
