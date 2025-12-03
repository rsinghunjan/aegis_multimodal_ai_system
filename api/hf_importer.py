a
"""
Hugging Face Hub importer for Aegis.

Features:
- Download a model repo from Hugging Face Hub (snapshot_download).
- Package the repo into a single tar.gz artifact suitable for registry ingestion.
- Optionally upload artifact to S3 (AWS) if S3_BUCKET env is configured.
- Sign the artifact using api.model_signing.sign_model_artifact (Vault Transit).
- Optionally register the artifact in the ModelRegistry via registry.register() using ModelConfig.

Usage:
  from api.hf_importer import import_from_hf
  import_from_hf(repo_id="google/flan-t5-small", model_name="flan-t5-small", version="hf-20251203",
                 sign_key="aegis-model-sign", upload_s3=True, register=True, registry=registry)

Notes:
- This script does not attempt to convert model weights to a particular runtime (ONNX/TorchScript).
  For conversion, set convert_to_torchscript=True and ensure torch is installed. Conversion is best-effort
  and may require custom model classes / example scripts.
- If you upload to S3, provide AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY and optionally AWS_REGION.
- The ModelRegistry.register() call requires ModelConfig and a storage-accessible model_path (s3://... or file path).
"""

import os
