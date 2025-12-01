import pytest
from aegis_multimodal_ai_system import error_handling as eh

def test_custom_exceptions():
    with pytest.raises(eh.AegisError):
        raise eh.AegisError("oops")

    with pytest.raises(eh.ModelLoadingError):
        raise eh.ModelLoadingError("model failed")  
