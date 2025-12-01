import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_FILE = LOG_DIR / "safety_audit.log"

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    h.setFormatter(formatter)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def audit_event(event: Dict[str, Any]) -> None:
    """
    Write a single JSON-line audit event to the audit log file and to the app logger.

    Event should be a JSON-serializable dict. Avoid writing full user content — instead include
    text_snippet or masked content as appropriate.
    """
    try:
        # Ensure a deterministic minimal representation
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        # Append to file (newline-delimited JSON)
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + os.linesep)
        # Also emit to the structured logger for easy integration with log collectors
        logger.info("AUDIT %s", line)
    except Exception as e:
        # Never crash the main flow due to audit logging failure; log locally
        logger.exception("Failed to write audit event: %s", e)
 
