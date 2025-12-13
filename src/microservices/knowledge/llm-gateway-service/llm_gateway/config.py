# llm_gateway/config.py

from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import List ,  Optional
import yaml
import boto3
from urllib.parse import urlparse
import os

class ModelConfig(BaseModel):
    """Configuration for a single AI model."""
    id: str
    hf_repo: str
    revision: str
    quantization: Optional[str] = None  # either a string or a null
    max_tokens: int 
    domain: str  
    role: str 

class LLMConfig(BaseModel):
    models: List[ModelConfig]

class Settings(BaseSettings):
    """
    Global application settings.
    Reads from environment variables automatically.
    """
    # Single URI for the registry
    MODELS_REGISTRY_S3_URI: str = "s3://neodustria-llm-catalog-staging/models.yaml"
    
    # Deployment settings
    VLLM_PORT: int = 8205
    USE_FAKE_LLM: bool = False  # Set to True for local dev or to use fake llm locally

    class Config:
        env_file = ".env"
        extra = "allow"   # it allows extra env variables and doesnot conflicts 

settings = Settings()

def load_config() -> LLMConfig:
    """
    Loads the model registry.
    - If S3 URI is present, tries to load from S3.
    - Falls back to local file if S3 fails or is not reachable.
    """
    uri = settings.MODELS_REGISTRY_S3_URI
    
    # Try S3 Loading
    if uri.startswith("s3://"):
        try:
            print(f"⬇️ Attempting to load config from S3: {uri}")
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            
            # Use Scaleway endpoint if configured
            endpoint = os.getenv("S3_ENDPOINT_URL", "https://s3.fr-par.scw.cloud")
            s3 = boto3.client("s3", endpoint_url=endpoint)
            
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = yaml.safe_load(obj["Body"].read())
            print("✅ Successfully loaded config from S3.")
            return LLMConfig(**data)
        except Exception as e:
            print(f"⚠️ S3 Load failed ({e}). Falling back to local models.yaml")

    # Local Fallback
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.yaml")
    with open(local_path, "r") as f:
        return LLMConfig(**yaml.safe_load(f))