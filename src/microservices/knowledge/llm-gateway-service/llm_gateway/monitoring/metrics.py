# llm_gateway/monitoring/metrics.py


from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

# --- METRIC NAMES ---

# Total number of requests
# Labels: 'model' (which AI), 'org' (which customer)
llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM requests processed",
    ["model", "org"],
)

# Latency distribution (how fast we answer)
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "Latency of LLM inference in seconds",
    ["model", "org"],
)

metrics_router = APIRouter()

@metrics_router.get("/metrics")
def metrics():
    """Exposes Prometheus metrics for scraping."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type="text/plain")