# Shared image for both the API and the Celery worker — they differ only by
# command, and keeping one image guarantees the worker's preprocessing matches
# what the API validated against.
#
# backend/ is self-contained (predict.py and data.py are vendored in — see
# backend/README.md on why), so build from backend/ itself:
#   docker build -f Dockerfile -t agrivert-backend .   (run from backend/)
#   docker build -f backend/Dockerfile -t agrivert-backend backend   (from repo root)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# opencv-python-headless still wants libGL/libglib at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/backend

# Torch first, from the CUDA index, as its own layer: it is by far the largest
# dependency and rarely changes.
# For a CPU-only deployment swap the index URL to .../whl/cpu and save ~2GB.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch>=2.4 torchvision>=0.19

COPY requirements.txt .
RUN pip install -r requirements.txt

# predict.py and data.py included via this COPY — they live in backend/
# alongside app/, not at a separate repo root.
COPY . .

ENV MODEL_REGISTRY_DIR=/srv/backend/models \
    PYTHONPATH=/srv/backend

EXPOSE 8000

# Overridden by docker-compose for the worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
