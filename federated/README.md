```markdown
Federated learning demo (local toy example)

This directory contains a minimal Flower (flwr) demo using scikit-learn models and synthetic data.
It is intended to be run locally for experimentation, CI smoke tests, or as a starting point for
adding secure aggregation and privacy controls.

Requirements
- Python environment with the project's requirements installed (see aegis_multimodal_ai_system/requirements.txt)
- Network ports: by default the Flower server runs on 8080.

Quick run (two terminals)
1) Start the server:
   python -m aegis_multimodal_ai_system.federated.server

2) Start one or more clients (in separate terminals):
   python -m aegis_multimodal_ai_system.federated.client --cid 1
   python -m aegis_multimodal_ai_system.federated.client --cid 2

Notes
- This demo uses sklearn's SGDClassifier with partial_fit on toy classification data generated per client.
- It's a toy simulation and does not implement production secure aggregation / differential privacy.
- To extend:
  - Integrate secure aggregation (e.g., add a secure aggregation strategy or use a library that supports it).
  - Add differential privacy (Opacus/TF Privacy) to client updates.
  - Replace sklearn model with a PyTorch/TensorFlow model for more complex experiments.
```
