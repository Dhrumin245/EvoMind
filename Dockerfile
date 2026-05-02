# escape=`
ARG PYTHON_IMAGE=python:3.13.3-windowsservercore-ltsc2022

FROM ${PYTHON_IMAGE} AS build

SHELL ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

ENV PYTHONDONTWRITEBYTECODE=1 `
    PYTHONUNBUFFERED=1 `
    PIP_NO_CACHE_DIR=1 `
    MPLBACKEND=Agg `
    SDL_VIDEODRIVER=dummy `
    SDL_AUDIODRIVER=dummy

WORKDIR C:\app

RUN python -m venv C:\venv

COPY requirements.txt .

RUN C:\venv\Scripts\python.exe -m pip wheel --wheel-dir C:\wheels -r requirements.txt

FROM ${PYTHON_IMAGE} AS runtime

SHELL ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

ENV PYTHONDONTWRITEBYTECODE=1 `
    PYTHONUNBUFFERED=1 `
    PIP_NO_CACHE_DIR=1 `
    MPLBACKEND=Agg `
    SDL_VIDEODRIVER=dummy `
    SDL_AUDIODRIVER=dummy

WORKDIR C:\app

RUN python -m venv C:\venv

COPY --from=build C:\wheels C:\wheels
COPY requirements.txt .

RUN C:\venv\Scripts\python.exe -m pip install --no-index --find-links=C:\wheels -r requirements.txt; `
    Remove-Item -Recurse -Force C:\wheels

COPY . .

RUN New-Item -ItemType Directory -Force C:\app\data, C:\app\artifacts, C:\app\backups | Out-Null; `
    icacls C:\app /grant 'Users:(OI)(CI)F' /T | Out-Null

USER ContainerUser

FROM runtime AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 `
  CMD ["powershell", "-NoProfile", "-Command", "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health/readiness' -TimeoutSec 5; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"]

CMD ["C:\\venv\\Scripts\\python.exe", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS worker

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 `
  CMD ["C:\\venv\\Scripts\\python.exe", "-m", "api.worker_healthcheck"]

CMD ["C:\\venv\\Scripts\\python.exe", "-m", "api.worker"]
