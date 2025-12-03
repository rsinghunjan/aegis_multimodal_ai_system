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
```markdown
MLflow integration for Aegis — quickstart

This folder provides a minimal MLflow stack (Postgres + MinIO + MLflow server) and example scripts
to instrument training runs and register artifacts into Aegis.

Prerequisites
- Docker & Docker Compose (v2 plugin preferred)
- Python 3.9+
- If you want to sign artifacts in CI/runtime, configure Vault (VAULT_ADDR + VAULT_TOKEN or OIDC login)

Start the MLflow stack (local dev)
1. From repo root:
   docker compose -f docker/docker-compose.mlflow.yml up -d

2. Wait for services:
   - MLflow UI at http://localhost:5000
   - MinIO console (if needed) at http://localhost:9000 (user/minioadmin)

Environment variables for training scripts
You can point training scripts to the MLflow server by setting:
- MLFLOW_TRACKING_URI (e.g. http://localhost:5000)
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for MinIO usage)
- MLFLOW_S3_ENDPOINT_URL (e.g. http://localhost:9000)

Example training & registration workflow
1. Run example training script:
   python examples/train_with_mlflow.py --experiment demo-mlflow --run-name trial1

   The script will:
   - log params/metrics via MLflow
   - log a model artifact (model.joblib) into MLflow artifact store

2. Register model artifact with Aegis (optional):
   python scripts/register_model_from_mlflow.py --run-id <RUN_ID> --artifact-path model.joblib --model-name my-model --sign-key aegis-model-sign

   Notes:
   - register script will attempt to sign artifact using Vault Transit via api.model_signing.sign_model_artifact().
   - It will attempt to call `api.registry.register()` if that API exists in your repo. If your registry API differs, adapt the script.

CI example
A sample GitHub Actions workflow (.github/workflows/mlflow-training.yml) demonstrates:
- starting the mlflow stack in the job
- running the training script and storing run/run_id as job output
- calling the registration script (requires Vault OIDC action to obtain ephemeral VAULT_TOKEN if signing)

Next steps
- Integrate MLflow runs with your ModelRegistry by storing run_id and provenance metadata.
- Add CI gating: require model validation (evaluation job) before promoting a model version.
- Replace local MinIO with your shared artifact bucket (S3) for team usage.

```
mlflow/README.md
