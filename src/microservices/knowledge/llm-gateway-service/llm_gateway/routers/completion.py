# llm_gateway/routers/completion.py

import time
import uuid
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from llm_gateway.models import generate
from llm_gateway.monitoring.metrics import llm_requests_total, llm_latency_seconds

router = APIRouter()

class CompletionRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 512

class CompletionResponse(BaseModel):
    request_id: str
    output: str
    model_id: str

@router.post("/vllm/completion", response_model=CompletionResponse)
def completion(req: CompletionRequest, x_org_id: str = Header("unknown")):
    request_id = str(uuid.uuid4())
    start = time.time()

    try:
        # Call the model logic (Fake or Real)
        output = generate(req.model_id, req.prompt, req.max_tokens)
    except StopIteration:
        raise HTTPException(status_code=404, detail=f"Unknown model_id={req.model_id}")

    # Record Metrics 
    duration = time.time() - start
    llm_requests_total.labels(model=req.model_id, org=x_org_id).inc()
    llm_latency_seconds.labels(model=req.model_id, org=x_org_id).observe(duration)

    return CompletionResponse(
        request_id=request_id,
        output=output,
        model_id=req.model_id
    )