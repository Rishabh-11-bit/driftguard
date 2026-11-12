# DriftGuard Guard API - self-hostable image.
# Build:  docker build -t driftguard .
# Run:    docker run -p 127.0.0.1:8080:8080 -e DRIFTGUARD_API_KEYS="<key>:<tenant>" \
#           -e DRIFTGUARD_REVIEWER_KEYS="<other key>:<tenant>" -e DRIFTGUARD_MODEL_KEY="<hmac key>" \
#           -v $PWD/models:/app/models:ro driftguard
# The models/ mount is REQUIRED (the image ships no model). Keep ONE worker: tenant state is in memory.
FROM python:3.12-slim
RUN useradd -m -u 10001 app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY driftguard ./driftguard
COPY service ./service
RUN pip install --no-cache-dir ".[service]"
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["uvicorn", "--factory", "service.api:create_app", "--host", "0.0.0.0", "--port", "8080", "--no-server-header"]
