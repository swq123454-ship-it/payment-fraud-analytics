FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so that a source change does not invalidate the layer.
# requirements.txt is installed whole, which includes the test and lint tooling.
# For an image that is built on every commit rather than every release, one
# dependency list that cannot drift out of step with CI is worth more than the
# few megabytes a production-only list would save.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# The scoring bundle is produced by the training pipeline and mounted or copied
# in at deploy time. The image never trains.
VOLUME ["/app/outputs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
