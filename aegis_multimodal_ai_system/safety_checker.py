import logging

logger = logging.getLogger(__name__)

class SafetyChecker:
    def __init__(self):
        pass  # Load your safety model here

    def is_unsafe(self, text: str, threshold: float = 0.7) -> bool:
        # Dummy logic: always return False (safe)
        return False
