# Backend image: FastAPI serving layer + CT orchestrator + ingestion scheduler
# (src/serving/api.py's lifespan runs all three in one process, by design —
# see CLAUDE.md §6). Training happens in-process on CPU (§9: CPU beats MPS/GPU
# ~12x for this 14.7K-parameter policy), so no CUDA wheels are installed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only torch first, from PyTorch's CPU index: the default PyPI wheel on
# linux/amd64 pulls ~2GB of CUDA libraries this service never uses.
COPY requirements.txt constraints.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" \
 && pip install -r requirements.txt -c constraints.txt

COPY src ./src
COPY scripts ./scripts
COPY .importlinter ./

# MLflow's tracking DB and artifacts live on one volume so a container
# replacement (every deploy) keeps every registered model — DR-09/NFR-07.
# Artifacts default to ./mlruns relative to WORKDIR, i.e. /app/mlruns.
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlruns.db
RUN useradd --create-home --uid 10001 app && mkdir -p /app/mlruns && chown -R app /app
USER app
VOLUME ["/app/mlruns"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
