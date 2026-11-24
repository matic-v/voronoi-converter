#!/usr/bin/env sh
# Starts the Voronoi Body Generator (installs dependencies on first run).
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
exec "$PY" scripts/start.py "$@"
