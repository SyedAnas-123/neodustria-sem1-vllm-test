# # 1. Use the pre-built vLLM image (Faster & Stable)
# FROM vllm/vllm-openai:latest

# # 2. Set working directory
# WORKDIR /app

# # 3. Install system tools (curl is useful for testing)
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# # 4. Copy requirements (for pydantic-settings, boto3, etc.)
# COPY requirements.txt .
# # vLLM is already installed, so we only install the extra stuff
# RUN pip install --no-cache-dir -r requirements.txt

# # 5. Copy your application code
# COPY llm_gateway ./llm_gateway

# # 6. CRITICAL: Copy the models.yaml file so the app can find it
# COPY models.yaml .

# # 7. Expose the port
# EXPOSE 8205

# # 8. Reset the entrypoint to allow custom commands
# ENTRYPOINT []

# # 9. Run the app
# CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8205"]

# 1. Use pre-built vLLM image
FROM vllm/vllm-openai:latest

# 2. Working directory
WORKDIR /app

# 3. System utilities
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 4. Install only extra Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY llm_gateway ./llm_gateway

# 6. Expose API port
EXPOSE 8205

# 7. Reset entrypoint (important)
ENTRYPOINT []

# 8. Run FastAPI app
CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8205"]
