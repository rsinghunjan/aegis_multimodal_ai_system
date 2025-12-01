"""
Secure enrollment endpoint for federated clients.

Hardening applied to address vulnerabilities:
- Reject API keys supplied in JSON body to avoid accidental logging/exposure; require x-api-key header.
- Limit request payload size (Content-Length) to mitigate large-body DOS.
- Create logs directory with restrictive permissions (0o700).
- Persist enrolled_clients.json atomically using tempfile.NamedTemporaryFile and os.replace,
  ensuring the temporary file is created in the same directory and is written with restrictive perms (0o600).
- Refuse to operate if the enrolled file path is a symlink to mitigate symlink-based tampering.
- Constant-time API key comparison (hmac.compare_digest) to mitigate timing attacks.
- Validate CID length and sanitize input.
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

from fastapi import Body, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, validator

import hmac

# Configuration
LOGS = Path("logs")
LOGS.mkdir(parents=True, exist_ok=True)
# Ensure restrictive permissions on the logs directory (owner rwx only)
try:
    os.chmod(LOGS, 0o700)
except Exception:
    # If chmod fails (e.g., permission issues on some platforms), continue but log
    logging.getLogger(__name__).warning("Could not enforce permissions on logs directory")

ENROLLED_FILE = LOGS / "enrolled_clients.json"

app = FastAPI(title="Aegis Federated Enrollment Service")
logger = logging.getLogger(__name__)

ALLOWED_CLIENT_API_KEYS = [k.strip() for k in os.getenv("ALLOWED_CLIENT_API_KEYS", "").split(",") if k.strip()]
_ALLOWED_KEYS_SET = set(ALLOWED_CLIENT_API_KEYS)


class EnrollRequest(BaseModel):
    cid: str = Field(..., description="Client id (string or integer, max length enforced)")
    # api_key is intentionally kept for backward-compatibility in schema but will be rejected at runtime
    api_key: str = Field(None, description="Client API key (DO NOT send in body; use x-api-key header)")

    @validator("cid")
    def cid_length(cls, v):
        s = str(v)
        if len(s) > 128:
            raise ValueError("cid too long")
        # basic sanitation: no control chars
        if any(ord(ch) < 32 for ch in s):
            raise ValueError("cid contains invalid characters")
        return s


def _read_enrolled() -> Dict:
    if ENROLLED_FILE.exists():
        try:
            # refuse to read if ENROLLED_FILE is a symlink (mitigate symlink attacks)
            if ENROLLED_FILE.is_symlink():
                logger.error("Enrolled file is a symlink; refusing to operate for safety")
                return {}
            return json.loads(ENROLLED_FILE.read_text(encoding="utf-8"))
        except Exception:
            # If file corrupted, return empty dict but log for operator attention
            logger.exception("Failed to parse enrolled clients file; starting fresh")
            return {}
    return {}


def _write_enrolled_atomic(data: Dict) -> None:
    """
    Atomically write enrolled data to ENROLLED_FILE using tempfile in the same directory + os.replace.

    Ensures the temporary file is created with restrictive permissions and fsynced before replace.
    """
    # Refuse to operate if ENROLLED_FILE exists and is a symlink (prevent TOCTOU via symlink)
    if ENROLLED_FILE.exists() and ENROLLED_FILE.is_symlink():
        logger.error("Refusing to overwrite enrolled file because it is a symlink")
        raise RuntimeError("enrolled file symlink detected")

    tmp = None
    try:
        # Create a named temporary file in the LOGS directory so os.replace is atomic within same filesystem
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=str(LOGS), prefix="enrolled_") as tf:
            tmp = Path(tf.name)
            # Write JSON deterministically
            json.dump(data, tf, indent=2, ensure_ascii=False, sort_keys=True)
            tf.flush()
            os.fsync(tf.fileno())

        # Ensure restrictive permissions on temp file (owner rw only)
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            logger.warning("Could not chmod temp enrolled file; continuing")

        # Atomic replace
        os.replace(str(tmp), str(ENROLLED_FILE))

        # Ensure final file has restrictive permissions
        try:
            os.chmod(ENROLLED_FILE, 0o600)
        except Exception:
            logger.warning("Could not chmod enrolled file; continuing")
    except Exception:
        # Attempt cleanup of temp file if it exists
        try:
            if tmp and tmp.exists():
                tmp.unlink()
        except Exception:
            logger.exception("Failed to remove temporary enrolled file")
        logger.exception("Failed to write enrollment file atomically")
        raise


def _is_api_key_allowed(provided_key: str) -> bool:
    """
    Constant-time comparison against allowed keys.
    """
    if not provided_key or not _ALLOWED_KEYS_SET:
        return False
    for allowed in _ALLOWED_KEYS_SET:
        if hmac.compare_digest(provided_key, allowed):
            return True
    return False


@app.post("/enroll")
async def enroll(
    request: Request,
    payload: EnrollRequest = Body(...),
    x_api_key: str = Header(None, alias="x-api-key"),
):
    """
    Enrollment endpoint.

    Security decisions:
    - API key MUST be provided via x-api-key header. Supplying api_key in JSON body is rejected.
    - Reject requests with Content-Length larger than allowed threshold to avoid large-body DOS.
    - Persist enrollment atomically with secure file permissions.
    """
    # Limit request body size (only enforce when Content-Length header is present)
    MAX_CONTENT_LENGTH = 4 * 1024  # 4 KiB
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_CONTENT_LENGTH:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Payload too large")
        except ValueError:
            # Malformed header -> reject
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Length header")

    # Disallow API key in request body to prevent accidental logging or exposure
    if payload.api_key:
        logger.warning("API key provided in body; rejecting for security reasons")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sending api_key in request body is forbidden; provide x-api-key header instead",
        )

    # Determine API key: must be present in header
    provided_key = None
    if x_api_key:
        provided_key = x_api_key.strip()

    if not _ALLOWED_KEYS_SET:
        # Enrollment disabled by operator (no allowed keys configured)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Enrollment disabled: server not configured to accept client keys",
        )

    if not provided_key or not _is_api_key_allowed(provided_key):
        # Do not reveal whether key is missing or invalid
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    # Validate CID
    cid = str(payload.cid).strip()
    if not cid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="cid is required")

    # Record enrollment in-memory then atomically persist
    enrolled = _read_enrolled()
    enrolled[cid] = {"cid": cid, "enrolled_at": __import__("time").time()}
    try:
        _write_enrolled_atomic(enrolled)
    except Exception:
        # If persistence fails, do not consider enrollment successful
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist enrollment")

    logger.info("Client enrolled: cid=%s", cid)
    return {"status": "enrolled", "cid": cid}
