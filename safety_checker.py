  
 
 
 

import logging
import re
from typing import Callable, Iterable, List, Optional

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


class SafetyChecker:
    """
    Lightweight safety checker with:
    - Optional injected model callable: model(text)->float (score 0.0..1.0, higher means more unsafe)
    - Keyword-based heuristics
    - PII regex detection (SSN as an example)
    """

    def __init__(
        self,
        blocked_keywords: Optional[Iterable[str]] = None,
        pii_regexes: Optional[List[re.Pattern]] = None,
        model_callable: Optional[Callable[[str], float]] = None,
    ):
        self.blocked_keywords = [
            k.lower() for k in (blocked_keywords or DEFAULT_BLOCKED_KEYWORDS)
        ]
        self.model_callable = model_callable
        # Basic PII regexes (example: US SSN); expand for production
        self.pii_regexes = pii_regexes or [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN 123-45-6789
            re.compile(r"\b\d{9}\b"),  # 9 digit straight
        ]
        logger.debug("SafetyChecker initialized with %d keywords", len(self.blocked_keywords))

    def is_unsafe(self, text: str, threshold: float = 0.7) -> bool:
        """
        Returns True if text is considered unsafe.
        Order of checks:
        1) If a model_callable is provided, use it and threshold.
        2) Keyword heuristic (case-insensitive substring match).
        3) PII regex match.
        """
        if not text:
            logger.debug("is_unsafe called with empty text -> safe")
            return False

        text_lower = text.lower()

        # 1) Model-based check (if provided)
        if self.model_callable is not None:
            try:
                score = float(self.model_callable(text))
                logger.debug("Model returned safety score %s for text: %r", score, text_lower[:80])
                if score >= threshold:
                    logger.info("Text flagged unsafe by model (score=%s)", score)
                    return True
            except Exception as e:
                logger.exception("Safety model callable raised an exception: %s", e)
                # Fall back to heuristics

        # 2) Keyword heuristic
        for kw in self.blocked_keywords:
            if kw and kw in text_lower:
                logger.info("Text flagged unsafe by keyword match: %s", kw)
                return True

        # 3) PII detection
        for regex in self.pii_regexes:
            if regex.search(text):
                logger.info("Text flagged unsafe by PII regex: %s", regex.pattern)
                return True

        logger.debug("Text considered safe by heuristics")
        return False
aegis_multimodal_ai_system/safety_checker.py
