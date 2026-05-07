FROM python:3.12-slim

WORKDIR /app

# Install only core dependencies first (layer cache)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[ui]"

# Copy source
COPY manifold/ ./manifold/
COPY manifold-ts/ ./manifold-ts/
COPY app.py deploy_shadow.py ./
COPY scripts/ ./scripts/

# Default: run the HTTP oracle server
EXPOSE 8080
ENV MANIFOLD_API_KEY=""
ENV MANIFOLD_DB_URL=""

CMD ["python", "-m", "manifold.server", "--port", "8080"]
