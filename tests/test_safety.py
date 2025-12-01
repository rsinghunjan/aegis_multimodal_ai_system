import pytest
from aegis_multimodal_ai_system.safety_checker import SafetyChecker

def test_safe_text():
    sc = SafetyChecker()
    assert sc.is_unsafe("Hello, how are you?") is False

def test_keyword_flagged():
    sc = SafetyChecker()
    assert sc.is_unsafe("I will kill you") is True

def test_pii_flagged():
    sc = SafetyChecker()
    assert sc.is_unsafe("My SSN is 123-45-6789") is True

def test_model_callable_used():
    # model returns 0.9 for anything containing "dangerous-model"
    def fake_model(text):
        return 0.9 if "dangerous-model" in text else 0.0

    sc = SafetyChecker(model_callable=fake_model)
    assert sc.is_unsafe("this text mentions dangerous-model", threshold=0.8) is True
    assert sc.is_unsafe("safe text", threshold=0.8) is False
