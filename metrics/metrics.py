from prometheus_client import Counter, Histogram, start_http_server

# Metrics exported on / by default when using prometheus_client.start_http_server
# Counters for safety flags with labels to indicate reason and detector
SAFETY_FLAG_COUNTER = Counter(
    "aegis_safety_flags_total",
    "Total number of times the safety checker flagged input",
    ["reason", "detected_by"],
)

# Histogram for latency of safety checks (seconds)
SAFETY_LATENCY_HISTOGRAM = Histogram(
    "aegis_safety_latency_seconds",
    "Latency in seconds for the safety checker",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0),
)


def start_metrics_server(port: int = 8000) -> None:
    """
    Starts a prometheus_client HTTP server in a background thread to serve metrics.
    Call this once at application startup.
    """
    start_http_server(port)
