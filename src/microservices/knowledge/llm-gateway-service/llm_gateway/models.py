# llm_gateway/models.py

# llm_gateway/models.py

from functools import lru_cache
from .config import load_config, ModelConfig, settings

# Load catalog on startup
_CATALOG = load_config()

# Only import real vLLM if we are NOT in fake mode
if not settings.USE_FAKE_LLM:
    from vllm import LLM, SamplingParams

@lru_cache(maxsize=16)
def get_llm(model_id: str):
    """
    Retrieves the LLM engine for a specific model ID.
    """
    # 1. Find the model config
    try:
        cfg = next(m for m in _CATALOG.models if m.id == model_id)
    except StopIteration:
        raise StopIteration(f"Model {model_id} not found in catalog")

    # 2. If Fake Mode, return nothing (bypass heavy load)
    if settings.USE_FAKE_LLM:
        return None, cfg

    # 3. Real Mode: Initialize vLLM
    # Note: On the GPU server, this will download full-precision weights!
    llm_kwargs = {
        "model": cfg.hf_repo,
        "revision": cfg.revision,
        "trust_remote_code": True,
    }

    # Quantization temporarily disabled to avoid vLLM AWQ config error
    # if getattr(cfg, "quantization", None):
    #     llm_kwargs["quantization"] = cfg.quantization

    llm = LLM(**llm_kwargs)
    return llm, cfg

def generate(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    """
    Generates text. Swaps logic based on USE_FAKE_LLM flag.
    """
    llm, cfg = get_llm(model_id)
    limit_tokens = min(max_tokens, cfg.max_tokens)

    # --- MOCK MODE ---
    if settings.USE_FAKE_LLM:
        return f"[FAKE RESPONSE] Model: {model_id} | Prompt: {prompt[:50]}..."

    # --- REAL MODE (GPU) ---
    params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=limit_tokens,
    )
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text
