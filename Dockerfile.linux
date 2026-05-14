FROM python:3.13-slim-bookworm AS runtime

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-compile --root-user-action=ignore --index-url "${PYTORCH_INDEX_URL}" torch==2.11.0 \
    && python -m pip install --no-compile --root-user-action=ignore -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/artifacts /app/backups \
    && useradd --create-home --shell /usr/sbin/nologin evomind \
    && chown -R evomind:evomind /app

USER evomind

FROM runtime AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/readiness', timeout=5).read()"]

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS worker

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "api.worker_healthcheck"]

CMD ["python", "-m", "api.worker"]
