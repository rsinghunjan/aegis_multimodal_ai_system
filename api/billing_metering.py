  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
"""
Billing metering hooks for hosted inference calls.

This module offers a simple function billing_meter_hf_call which records a usage event.
Integrate with your billing collector / event pipeline to persist invoiceable usage.

- For simple deployments you can write a DB row into usage_events table.
- For high throughput, send events to a metrics stream / Kafka and run a billing batcher.

Provided: a minimal synchronous implementation that writes to SQL via your existing SessionLocal.
"""
import os
import logging
import time
from typing import Optional

try:
    from prometheus_client import Counter, Histogram
    METRICS_AVAILABLE = True
    HF_USAGE_COUNTER = Counter("aegis_hf_usage_total", "Hosted HF inference calls", ["model"])
    HF_USAGE_DURATION = Histogram("aegis_hf_usage_duration_seconds", "Duration of HF inference calls", ["model"])
except Exception:
    METRICS_AVAILABLE = False

from api.db import SessionLocal
from api.models import UsageEvent  # you should define a UsageEvent model / table

logger = logging.getLogger("aegis.billing_metering")


def billing_meter_hf_call(model: str, tenant_id: Optional[str], duration_s: float, request_bytes: int, response_bytes: int):
    """
    Record a single usage event for billing / quota enforcement.

    Implementations should be idempotent and low-latency. This simple version writes a DB row.
    """
    try:
        if METRICS_AVAILABLE:
            HF_USAGE_COUNTER.labels(model=model).inc()
            HF_USAGE_DURATION.labels(model=model).observe(duration_s)
    except Exception:
        logger.exception("prometheus emission failed")

    # Minimal DB write: UsageEvent(table) should be created in api.models with fields:
    # tenant_id, provider, model, duration_s, request_bytes, response_bytes, created_at
    try:
        session = SessionLocal()
        ue = UsageEvent(
            tenant_id=tenant_id or "unknown",
            provider="huggingface",
            model=model,
            duration_seconds=float(duration_s),
            request_bytes=int(request_bytes),
            response_bytes=int(response_bytes),
        )
        session.add(ue)
        session.commit()
    except Exception:
        logger.exception("billing meter DB write failed (non-fatal)")
    finally:
        try:
            session.close()
        except Exception:
            pass
