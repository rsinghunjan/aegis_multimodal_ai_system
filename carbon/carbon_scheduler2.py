"""
Carbon-aware scheduler helper with small logging & thread-safe cache.
"""

import os
import time
import json
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE = {"ts": 0, "value": None}
_CACHE_LOCK = threading.Lock()

def _get_env(name: str, default: Optional[str]=None):
    return os.environ.get(name, default)

def _fetch_from_api(url: str, timeout: int = 5) -> Optional[float]:
    # Try requests first
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("requests fetch failed: %s", e)
        # Fallback to urllib
        try:
            from urllib.request import urlopen, Request
            req = Request(url, headers={"Accept":"application/json"})
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                data = json.loads(body.decode("utf-8"))
        except Exception as e2:
            logger.warning("failed to fetch carbon intensity from %s: %s", url, e2)
            return None

    # Expect a simple numeric field "carbon_intensity" or "data"."intensity"
    if isinstance(data, dict):
        if "carbon_intensity" in data:
            return float(data["carbon_intensity"])
        if "data" in data and isinstance(data["data"], dict) and "intensity" in data["data"]:
            return float(data["data"]["intensity"])
    logger.debug("unexpected carbon API response shape: %s", data)
    return None

def get_current_intensity() -> Optional[float]:
    """
    Returns the cached carbon intensity if within TTL, otherwise queries the API.
    """
    ttl = int(_get_env("CARBON_CACHE_TTL", "300"))
    now = int(time.time())
    with _CACHE_LOCK:
        if _CACHE["ts"] + ttl > now and _CACHE["value"] is not None:
            logger.debug("returning cached carbon intensity: %s", _CACHE["value"])
            return _CACHE["value"]

    url = _get_env("CARBON_API_URL", "")
    # Optional local override for testing
    local_override = _get_env("CARBON_LOCAL_INTENSITY")
    if local_override:
        try:
            value = float(local_override)
            with _CACHE_LOCK:
                _CACHE.update({"ts": now, "value": value})
            logger.debug("using local override carbon intensity: %s", value)
            return value
        except Exception:
            logger.exception("invalid CARBON_LOCAL_INTENSITY: %s", local_override)

    if not url:
        logger.debug("no CARBON_API_URL configured")
        return None

    value = _fetch_from_api(url)
    if value is not None:
        with _CACHE_LOCK:
            _CACHE.update({"ts": now, "value": value})
    return value

def should_schedule_now(threshold: float = 100.0) -> bool:
    """
    Return True if current intensity is below the threshold (or unknown).
    Conservative default: if unknown, returns True to avoid blocking.
    """
    val = get_current_intensity()
    if val is None:
        return True
    try:
        return float(val) <= float(threshold)
    except Exception:
        logger.exception("error comparing carbon intensity: %s", val)
        return True
