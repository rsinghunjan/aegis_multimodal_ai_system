"""
Carbon-aware scheduler helper.

- Queries a configurable carbon intensity API endpoint (env var: CARBON_API_URL)
- Caches the last value for a TTL (env var: CARBON_CACHE_TTL, default 300s)
- Exposes:
    get_current_intensity() -> float | None
    should_schedule_now(threshold: float) -> bool

Dependency-light:
- Uses requests if available, otherwise falls back to urllib.
- No external scheduler integration here; this is a helper to call from job schedulers.
"""

import os
import time
import json
from typing import Optional

_CACHE = {"ts": 0, "value": None}

def _get_env(name: str, default: Optional[str]=None):
    return os.environ.get(name, default)

def _fetch_from_api(url: str, timeout: int = 5) -> Optional[float]:
    # Try requests first
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        # Fallback to urllib
        try:
            from urllib.request import urlopen, Request
            req = Request(url, headers={"Accept":"application/json"})
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                data = json.loads(body.decode("utf-8"))
        except Exception:
            return None

    # Expect a simple numeric field "carbon_intensity" or "data"."intensity"
    if isinstance(data, dict):
        if "carbon_intensity" in data:
            return float(data["carbon_intensity"])
        if "data" in data and isinstance(data["data"], dict) and "intensity" in data["data"]:
            return float(data["data"]["intensity"])
    return None

def get_current_intensity() -> Optional[float]:
    """
    Returns the cached carbon intensity if within TTL, otherwise queries the API.
    """
    global _CACHE
    ttl = int(_get_env("CARBON_CACHE_TTL", "300"))
    now = int(time.time())
    if _CACHE["ts"] + ttl > now and _CACHE["value"] is not None:
        return _CACHE["value"]

    url = _get_env("CARBON_API_URL", "")
    # Optional local override for testing
    local_override = _get_env("CARBON_LOCAL_INTENSITY")
    if local_override:
        try:
            value = float(local_override)
            _CACHE.update({"ts": now, "value": value})
            return value
        except Exception:
            pass

    if not url:
        return None

    value = _fetch_from_api(url)
    if value is not None:
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
        return True
