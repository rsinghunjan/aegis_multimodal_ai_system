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
 65
#!/usr/bin/env python3
"""
Lightweight runtime enforcement patch.

When imported early in the application (e.g., in the WSGI/ASGI entrypoint),
this module will monkeypatch known loader entrypoints to ensure fetch_and_verify_model is called
before the model binary is loaded. This is intended as an incremental enforcement mechanism
so you can deploy quickly and then convert callers to the explicit fetch_and_prepare_model_for_load.

Usage:
  # early in app startup:
  import aegis_multimodal_ai_system.orchestrator.enforce_verification_patch as ev
  ev.patch_loaders()
"""
from __future__ import annotations
import logging
from functools import wraps
from typing import Callable, Any

from aegis_multimodal_ai_system.model_registry.verify_and_download import fetch_and_verify_model

LOG = logging.getLogger("aegis.enforce_verify")

def _wrap_loader(fn: Callable[..., Any], model_name_arg_index: int = 0, artifact_name: str = "model.onnx"):
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        try:
            # attempt to extract model_name argument
            if len(args) > model_name_arg_index:
                model_name = args[model_name_arg_index]
            else:
                model_name = kwargs.get("model_name")
            if model_name:
                LOG.debug("Enforcing verification for model %s", model_name)
                # will raise on verification failure
                local_path = fetch_and_verify_model(str(model_name), artifact_name=artifact_name)
                # inject resolved local path into kwargs as 'local_artifact' if caller accepts it
                kwargs.setdefault("local_artifact", local_path)
        except Exception as e:
            LOG.exception("Model verification failed; refusing to load: %s", e)
            raise
        return fn(*args, **kwargs)
    return _wrapped

def patch_loaders():
    """
    Patch commonly used loader functions. Add more as you identify them.
    NOTE: patching is optional and must be applied early.
    """
    try:
        # Example: patch TF loader if present
        import aegis_multimodal_ai_system.loaders.tf_loader as tf_loader  # may not exist
        if hasattr(tf_loader, "load_savedmodel"):
            tf_loader.load_savedmodel = _wrap_loader(tf_loader.load_savedmodel, model_name_arg_index=0, artifact_name="saved_model")
            LOG.info("Patched tf_loader.load_savedmodel to enforce verification")
    except Exception:
        LOG.debug("tf_loader not present or could not be patched")
    try:
        # Example: patch ONNX loader if present
        import aegis_multimodal_ai_system.loaders.onnx_loader as onnx_loader  # may not exist
        if hasattr(onnx_loader, "load_onnx_model"):
            onnx_loader.load_onnx_model = _wrap_loader(onnx_loader.load_onnx_model, model_name_arg_index=0, artifact_name="model.onnx")
            LOG.info("Patched onnx_loader.load_onnx_model to enforce verification")
    except Exception:
        LOG.debug("onnx_loader not present or could not be patched")
aegis_multimodal_ai_system/orchestrator/enforce_verification_patch.py
