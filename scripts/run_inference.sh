
#!/usr/bin/env bash
set -euo pipefail

# Simple local run command (assumes PYTHONPATH includes repo dir)
UVICORN_CMD=${UVICORN_CMD:-"uvicorn aegis_multimodal_ai_system.inference.server:app --host 0.0.0.0 --port 9000 --log-level info"}

echo "Starting inference server on :9000"
exec $UVICORN_CMD
