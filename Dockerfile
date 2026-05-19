FROM python:3.12-slim

WORKDIR /app

# Copy dependency manifests and all source before installing
# (editable install requires the source tree to be present)
COPY pyproject.toml requirements.txt ./
COPY manifold/ ./manifold/
COPY manifold_physical/ ./manifold_physical/
COPY manifold-ts/ ./manifold-ts/
COPY app.py deploy_shadow.py ./
COPY scripts/ ./scripts/

# Install the package with UI extras — deps resolved from pyproject.toml
RUN pip install --no-cache-dir -e ".[ui]"

# Default: run the HTTP oracle server
EXPOSE 8080
ENV MANIFOLD_API_KEY=""
ENV MANIFOLD_DB_URL=""

CMD ["python", "-m", "manifold.server", "--port", "8080"]
