#!/usr/bin/env bash
set -euo pipefail

# Start the federated server (blocking). Useful for local dev containers.
python -m aegis_multimodal_ai_system.federated.server
