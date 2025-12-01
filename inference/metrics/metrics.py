  
from prometheus_client import Counter, Histogram, start_http_server

# Existing safety metrics (left intact)
SAFETY_FLAG_COUNTER = Counter(
    "aegis_safety_flags_total",
    "Total number of times the safety checker flagged input",
    ["reason", "detected_by"],
)

SAFETY_LATENCY_HISTOGRAM = Histogram(
    "aegis_safety_latency_seconds",
    "Latency in seconds for the safety checker",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0),
)

# New inference metrics
INFERENCE_REQUESTS_TOTAL = Counter(
    "aegis_inference_requests_total",
    "Total number of inference requests received",
    ["model_version"],
)

INFERENCE_PREDICTIONS_TOTAL = Counter(
    "aegis_inference_predictions_total",
    "Total number of output predictions generated",
    ["model_version"],
)

INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "aegis_inference_latency_seconds",
    "Latency in seconds for the inference request (end-to-end)",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0),
)


def start_metrics_server(port: int = 8000) -> None:
    """
    Starts a prometheus_client HTTP server in a background thread to serve metrics.
    Call this once at application startup.
    """
    start_http_server(port)
