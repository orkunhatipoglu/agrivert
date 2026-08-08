# Shared image for both the API and the Celery worker — they differ only by
# command, and keeping one image guarantees the worker's preprocessing matches
# what the API validated against.
#
# The image needs both backend/ and the ml/ package (serving imports
# ml.predict), so build from the REPO ROOT:
#   docker build -f backend/Dockerfile -t agrivert-backend .

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

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# The shared training/serving code. Serving imports ml.predict, which imports
# ml.data, so preprocessing IS the training preprocessing — not a copy.
COPY ml/ /srv/ml/
COPY backend/ /srv/backend/

ENV MODEL_REGISTRY_DIR=/srv/backend/models \
    ML_REPO_ROOT=/srv \
    PYTHONPATH=/srv:/srv/backend

EXPOSE 8000

# Overridden by docker-compose for the worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
