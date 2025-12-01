import logging
import re
import time
import uuid
from typing import Callable, Iterable, List, Optional

from .audit.audit_logger import audit_event
from .metrics.metrics import SAFETY_FLAG_COUNTER, SAFETY_LATENCY_HISTOGRAM

logger = logging.getLogger(__name__)

DEFAULT_BLOCKED_KEYWORDS = [
    # Simple illustrative keyword list; replace with production-safe policies or model
    "kill",
    "murder",
    "suicide",
    "bomb",
    "porn",
    "child sexual",
    "ssn",
    "social security",
]
