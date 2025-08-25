Python 3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> # main.py
... from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
... from fastapi.middleware.cors import CORSMiddleware
... from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
... from pydantic import BaseModel, BaseSettings
... from redis import asyncio as aioredis
... import httpx
... import logging
... import time
... from typing import Optional, List
... import uuid
... 
... # --- Configuration (would use environment variables) ---
... class Settings(BaseSettings):
...     app_name: str = "Aegis Production API"
...     redis_url: str = "redis://localhost:6379"
...     vllm_url: str = "http://vllm-server:8000/v1"
...     safety_checker_url: str = "http://triton-server:8000"
...     api_keys: List[str] = ["your-secure-api-key-here"]  # Load from vault
... 
...     class Config:
...         env_file = ".env"
... 
... settings = Settings()
... 
... # --- Security ---
... security = HTTPBearer()
... 
... def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
...     if credentials.credentials not in settings.api_keys:
...         raise HTTPException(status_code=401, detail="Invalid API key")
...     return credentials.credentials
... 
... # --- App Setup ---
... app = FastAPI(title=settings.app_name)
... app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Connections (Async) ---
@app.on_event("startup")
async def startup_event():
    app.state.redis = await aioredis.from_url(settings.redis_url)
    app.state.http_client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()
    await app.state.http_client.accept()

# --- Rate Limiting Middleware ---
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    api_key = request.headers.get("Authorization")
    if api_key:
        # Use API key in rate limit calculation
        key = f"rate_limit:{api_key}"
    else:
        # Use IP address as a fallback
        key = f"rate_limit:{request.client.host}"

    # Allow 10 requests per minute per key/IP
    current = await app.state.redis.get(key)
    if current and int(current) > 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    await app.state.redis.incr(key, 1)
    await app.state.redis.expire(key, 60)
    response = await call_next(request)
    return response

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_safe: bool

# --- Core Endpoint ---
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    # 1. Check cache first (for idempotent requests)
    cache_key = f"chat:{request.message}"
    cached_response = await app.state.redis.get(cache_key)
    if cached_response:
        return ChatResponse(**json.loads(cached_response))

    # 2. Safety Check - Call external Triton server
    safety_payload = {"text": request.message}
    try:
        safety_response = await app.state.http_client.post(
            f"{settings.safety_checker_url}/safety",
            json=safety_payload,
            timeout=5.0
        )
        safety_result = safety_response.json()
        if safety_result.get("is_unsafe"):
            raise HTTPException(status_code=400, detail="Query violates safety policy.")
    except httpx.RequestError:
        # Log the error but proceed? Block? Depends on safety requirements.
        # In production, you might want to fail closed.
        logging.error("Safety service unavailable")

    # 3. Call vLLM server for the LLM
    llm_payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": request.message}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    try:
        llm_response = await app.state.http_client.post(
            f"{settings.vllm_url}/chat/completions",
            json=llm_payload,
            timeout=30.0
        )
        llm_result = llm_response.json()
        raw_response = llm_result['choices'][0]['message']['content']
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    # 4. Async Output Safety Check (in background to not block response)
    background_tasks.add_task(perform_async_safety_check, raw_response)

    # 5. Cache the response
    session_id = request.session_id or str(uuid.uuid4())
    final_response = ChatResponse(response=raw_response, session_id=session_id, is_safe=True)
    await app.state.redis.setex(cache_key, 300, json.dumps(final_response.dict())) # Cache for 5 minutes

    return final_response

async def perform_async_safety_check(text: str):
    # ... Async call to safety checker ...
    pass

if __name__ == "__main__":
    import uvicorn
