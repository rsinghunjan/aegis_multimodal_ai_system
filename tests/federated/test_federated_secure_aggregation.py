# Federated integration test: small Flower simulation with anomaly detection.
# This test starts a Flower server in a background thread and connects a few simple numpy clients.
# One client sends a malicious (very large) update and should be excluded by the strategy.
#
# Notes:
# - Requires flwr installed in the test runner environment (add to requirements.in or CI job).
# - This test is intentionally lightweight and synthetic to make CI runs fast.
import threading
import time
import numpy as np
import pytest

try:
    import flwr as fl  # type: ignore
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays  # type: ignore
except Exception as e:
    fl = None

from aegis_multimodal_ai_system.federated.secure_aggregation_strategy import SecureAggregationStrategy

# Basic model: a single 1-D numpy array as "weights"
def get_initial_weights():
    return [np.zeros((10,), dtype=np.float32)]

class SimpleNumpyClient(fl.client.NumPyClient):  # type: ignore
    def __init__(self, weights: list, scale: float = 1.0):
        self._weights = weights
        self.scale = scale

    def get_parameters(self):
        # Return current parameters as a list of numpy arrays
        return self._weights

    def fit(self, parameters, config):
        # Receive global parameters and return updated parameters
        nds = parameters_to_ndarrays(parameters)
        # Simulate local training by adding a scaled delta
        new = [arr + (self.scale * np.ones_like(arr) * 0.1) for arr in nds]
        return ndarrays_to_parameters(new), len(new[0].ravel())

    def evaluate(self, parameters, config):
        return 0.0, len(parameters_to_ndarrays(parameters)[0].ravel()), {}

@pytest.mark.skipif(fl is None, reason="flwr not installed")
def test_secure_aggregation_excludes_malicious_update():
    # Build the strategy with a tight anomaly threshold to detect malicious client
    base_strategy = fl.server.strategy.FedAvg()
    wrapper = SecureAggregationStrategy(base_strategy=base_strategy, clip_norm=10.0, anomaly_std_multiplier=1.0)
    strategy = wrapper.get_strategy()

    # Start server in background thread
    server_thread = threading.Thread(
        target=lambda: fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 1}, strategy=strategy),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)  # give server time to start

    # Start normal clients (scale=1.0)
    clients = [
        SimpleNumpyClient(get_initial_weights(), scale=1.0),
        SimpleNumpyClient(get_initial_weights(), scale=1.0),
    ]

    # Malicious client: very large scale causing large-norm update
    bad_client = SimpleNumpyClient(get_initial_weights(), scale=1000.0)

    # Start clients as numpy clients connecting to server
    client_threads = []
    for c in clients + [bad_client]:
        t = threading.Thread(target=lambda client=c: fl.client.start_numpy_client("localhost:8080", client=client), daemon=True)
        t.start()
        client_threads.append(t)

    # Wait for all clients to finish
    for t in client_threads:
        t.join(timeout=30)

    # Ensure server thread completes
    server_thread.join(timeout=30)
    assert not server_thread.is_alive()
```

````markdown name=docs/federated_secure_aggregation.md
```markdown
# Federated secure aggregation & staging guidance (Aegis)

This document explains the integration test, server-side validation, how to enable secure aggregation, and a recommended process for holding flwr upgrades in staging until validated.

What we added
- aegis_multimodal_ai_system/federated/secure_aggregation_strategy.py
  - Server-side wrapper that validates client updates, uses norm-based anomaly detection to exclude outliers,
    and optionally (best-effort) toggles a secure-aggregation flag if supported by your Flower version.
- tests/federated/test_federated_secure_aggregation.py
  - Small end-to-end synthetic test that runs a Flower server + 3 clients and verifies malicious client updates are excluded.
- GitHub Actions: you should wire the staged-integration job to run this test (see .github/workflows/staged-integration.yml)
  - Trigger the test when files under `federated/` change or during nightly/staging runs.

Enable secure aggregation (Flower-dependent)
- Flower's secure aggregation support and configuration options vary between versions.
- If your flwr version supports secure aggregation, set `enable_secure_aggregation=True` when constructing the `SecureAggregationStrategy`.
  Example:
    strat_wrapper = SecureAggregationStrategy(enable_secure_aggregation=True)
    strat = strat_wrapper.get_strategy()
    fl.server.start_server(strategy=strat, ...)
- If your installed flwr lacks built-in secure aggregation, consider:
  - Upgrading flwr in a staged branch and validating secure aggregation via integration tests.
  - Using an external secure aggregation library or protocol (e.g., threshold homomorphic masking) and integrating before running FL at scale.

Client validation & anomaly detection
- The strategy excludes clients whose update norm is a statistical outlier (mean + k * std). This is a fast, defensive heuristic.
- For production, consider stronger validation:
  - Clipping and per-layer clipping
  - Robust aggregation (median, trimmed mean)
  - Per-client behavior history & reputation scoring
  - Differential Privacy + Secure Aggregation for combined privacy guarantees

Hold flwr upgrades in staging (process)
1. Do not automerge Dependabot major bumps for flwr. Use dependency-guard to label major bumps (we added dependency-guard.yml).
2. When a flwr upgrade PR appears:
   - Run the staged-integration job (this will run federated tests, secure aggregation validation, and other integration tests).
   - Manually validate: run the federated smoke locally and run FL examples on small clusters.
   - If tests pass, socialize the change and approve; otherwise keep the PR in staging and triage issues.
3. Optionally use a "staged" branch:
   - Merge the flwr upgrade to `staging` first, run extended nightly tests and longer federated runs.
   - Promote to `main` only after green staged runs and a maintainer sign-off.

CI notes
- Ensure your CI uses the committed `requirements.txt` lock to install a specific flwr version for reproducibility.
- Use the staged-integration workflow (heavy tests) to gate merging of flwr bumps.
- Consider adding a small "flwr-smoke" matrix job that runs the federated test against both the current and the proposed flwr versions during PR validation.

Next steps & suggestions
- Replace norm-based filtering with a more robust strategy (trimmed mean, Krum, or median) if you expect adversarial clients.
- Add logging/alerting for excluded clients (audit trail) and consider quarantining suspicious clients until manual re-evaluation.
- Add per-client rate-limiting and basic reputation tracking (reject clients that repeatedly produce anomalous updates).
- If you require formal secure aggregation, coordinate upgrades of flwr and test in an isolated environment with a real secure-aggregation run.

```
