# llmobs server: API + viewer on uvicorn, SQLite in /data.
#   docker compose up -d --build        (see compose.yaml and the README's "Deploy" section)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LLMOBS_DB=/data/llmobs.db

WORKDIR /app
COPY pyproject.toml README.md ./
COPY llmobs ./llmobs
COPY llmobs_eval ./llmobs_eval
COPY llmobs_server ./llmobs_server
RUN pip install ".[server]" && rm -rf /app/*

# Unprivileged user that owns only the data directory.
RUN useradd --system --uid 10001 --no-create-home llmobs \
    && mkdir /data && chown llmobs /data
USER llmobs
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

# One worker: SQLite with a single shared connection is a one-process store.
CMD ["uvicorn", "llmobs_server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
