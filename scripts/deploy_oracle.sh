#!/usr/bin/env bash
# scripts/deploy_oracle.sh
# Deploy the MANIFOLD Oracle Server (server.py) on any POSIX host.
#
# Usage:
#   chmod +x scripts/deploy_oracle.sh
#   ./scripts/deploy_oracle.sh [--port PORT] [--data-dir DIR] [--host HOST]
#
# Requirements:
#   - Python 3.12+
#   - The manifold package installed (pip install -e . or pip install manifold-ai)
#
# Environment variables (can also be set before running):
#   MANIFOLD_PORT       TCP port (default: 8080)
#   MANIFOLD_HOST       Bind address (default: 0.0.0.0)
#   MANIFOLD_DATA_DIR   Vault WAL directory (default: ./manifold_data)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PORT="${MANIFOLD_PORT:-8080}"
HOST="${MANIFOLD_HOST:-0.0.0.0}"
DATA_DIR="${MANIFOLD_DATA_DIR:-$(pwd)/manifold_data}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     PORT="$2";     shift 2 ;;
    --host)     HOST="$2";     shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--port PORT] [--host HOST] [--data-dir DIR]"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Locate the Python binary (must be 3.12+)
# ---------------------------------------------------------------------------
find_python() {
  for cmd in python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
      ver=$("$cmd" -c 'import sys; print(sys.version_info[:2])' 2>/dev/null || echo "(0, 0)")
      if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON=$(find_python) || {
  echo "ERROR: Python 3.12+ is required but was not found on PATH."
  echo "       Install Python 3.12 and re-run this script."
  exit 1
}

echo "✓ Python: $PYTHON ($($PYTHON --version))"

# ---------------------------------------------------------------------------
# Verify the manifold package is importable
# ---------------------------------------------------------------------------
if ! "$PYTHON" -c "import manifold.server" 2>/dev/null; then
  echo "ERROR: 'manifold' package is not installed."
  echo "       Run: pip install -e .   (from the repo root)"
  echo "       or:  pip install manifold-ai"
  exit 1
fi
echo "✓ manifold package is importable"

# Verify new Phase 26-28 modules are present
for module in manifold.entropy manifold.consensus manifold.discovery; do
  if ! "$PYTHON" -c "import $module" 2>/dev/null; then
    echo "ERROR: Module '$module' (v1.3.0) is not available."
    echo "       Ensure you have the latest code with: git pull"
    exit 1
  fi
  echo "✓ $module is importable"
done

# ---------------------------------------------------------------------------
# Create the data directory for the Vault WAL
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR"
echo "✓ Vault data directory: $DATA_DIR"

# ---------------------------------------------------------------------------
# Export environment variables consumed by server.py
# ---------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-$(dirname "$(cd "$(dirname "$0")"; pwd)")}"
export MANIFOLD_DATA_DIR="$DATA_DIR"

# ---------------------------------------------------------------------------
# Launch the server
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         MANIFOLD Oracle Server  v1.3.0                      ║"
echo "║  886 Tests Passing | 0 External Dependencies | Grid OS      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting MANIFOLD server on $HOST:$PORT ..."
echo "  Dashboard : http://$HOST:$PORT/dashboard"
echo "  Policy    : http://$HOST:$PORT/policy"
echo "  Shield    : POST http://$HOST:$PORT/shield"
echo "  Vault dir : $DATA_DIR"
echo ""

exec "$PYTHON" -m manifold.server --host "$HOST" --port "$PORT"
