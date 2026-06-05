FROM python:3.11-slim

# libgomp1: OpenMP runtime required by LightGBM and XGBoost wheels
# curl: container healthcheck for the dashboard service
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install third-party dependencies in their own layer so editing source code
# does not re-download every wheel. tomllib reads the dep list straight from
# pyproject.toml — no duplicated requirements.txt to drift out of sync.
COPY pyproject.toml ./
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Install the package itself (deps already satisfied above)
COPY src ./src
COPY README.md ./
RUN pip install --no-cache-dir --no-deps -e .

COPY dashboard ./dashboard
COPY tests ./tests

# Artifact and data mount points, writable by the non-root user
RUN useradd --create-home appuser \
    && mkdir -p /app/dataset /app/data /app/models /app/reports \
    && chown -R appuser:appuser /app
USER appuser

ENV SMART_RETAIL_RAW_DIR=/app/dataset \
    SMART_RETAIL_DATA_DIR=/app/data \
    SMART_RETAIL_MODELS_DIR=/app/models \
    SMART_RETAIL_REPORTS_DIR=/app/reports \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "smart_retail.pipeline"]
