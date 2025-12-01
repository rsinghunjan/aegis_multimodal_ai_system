"""
Simple Flower (flwr) server wrapper.

This module tries to import the Flower package (flwr). If it's not installed,
it provides a helpful message that explains how to install it and run a
development federated server.

Notes:
- Keep the wrapper lightweight; do not require flwr at import-time in production code paths.
- For production, run the flwr server in a dedicated process (or k8s job) and secure communications.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def start_flower_server(server_config: Optional[dict] = None):
    """
    Start a Flower server if flwr is installed. server_config is a dict passed into flwr.server.start_server.
    Basic usage (if flwr installed):
        from aegis_multimodal_ai_system.federated.flower_server import start_flower_server
        start_flower_server({"server_address":"0.0.0.0:8080"})

    If flwr isn't installed, this function prints instructions.
    """
    try:
        import flwr as fl
    except Exception:
        logger.warning("Flower (flwr) is not installed. Install with `pip install flwr` to run federated server.")
        print("""
Flower (flwr) not found. To install and run a basic server:
  pip install flwr
  python -c "import flwr as fl; fl.server.start_server(server_address='0.0.0.0:8080')"

For production, run the server in its own container/process and secure channels (mTLS).
""")
        return

    cfg = server_config or {}
    server_address = cfg.get("server_address", "0.0.0.0:8080")
    logger.info(f"Starting Flower server at {server_address}")
    # Minimal start; for production pass strategy, config, and secure transport
    fl.server.start_server(server_address=server_address)
