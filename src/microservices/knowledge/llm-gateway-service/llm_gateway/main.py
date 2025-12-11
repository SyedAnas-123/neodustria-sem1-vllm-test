# llm_gateway/main.py

"""
Main FastAPI application for the LLM Gateway.
- Exposes /health
- Includes:
  - /v1/vllm/completion
  - /metrics
"""

from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv() # Load env vars immediately

from llm_gateway.routers import completion
from llm_gateway.monitoring.metrics import metrics_router

app = FastAPI(title="Neodustria LLM Gateway")

@app.get("/health")
def health():
    return {"status": "ok", "service": "llm-gateway-service"}

# Mount Routers
app.include_router(completion.router, prefix="/v1")
app.include_router(metrics_router, prefix="")
