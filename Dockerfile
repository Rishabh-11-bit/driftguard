# DriftGuard Guard API - self-hostable image.
# Build:  docker build -t driftguard .
# Run:    docker run -p 127.0.0.1:8080:8080 -e DRIFTGUARD_API_KEYS="<key>:<tenant>" \
#           -v $PWD/models:/app/models:ro driftguard
FROM python:3.12-slim
RUN useradd -m -u 10001 app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY driftguard ./driftguard
COPY service ./service
RUN pip install --no-cache-dir ".[service]"
USER app
EXPOSE 8080
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8080"]
