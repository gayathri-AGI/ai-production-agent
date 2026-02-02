"""
🚀 Production AI Agent API
   Ready for deployment to Render / Railway / Hugging Face

   Local run: uvicorn app:app --reload
   Production: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import json
import hashlib
import uuid
from datetime import datetime
from collections import deque
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache

# ============================================================
# CONFIGURATION (All from environment variables!)
# ============================================================

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
MODEL_NAME = os.environ.get('MODEL_NAME', 'groq/llama-3.1-8b-instant')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '512'))
RATE_LIMIT = int(os.environ.get('RATE_LIMIT', '30'))  # requests per minute

# Validate on startup
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set! API calls will fail.")
    print("   Set it in your environment or platform dashboard.")

# ============================================================
# STRUCTURED LOGGING
# ============================================================

def log(level: str, message: str, **kwargs):
    """Output structured JSON log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "service": "ai-agent-api",
        "message": message,
        **kwargs
    }
    print(json.dumps(entry))

# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = Lock()
    
    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            # Check limit
            if len(self.requests) >= self.max_requests:
                return False
            self.requests.append(now)
            return True
    
    def remaining(self) -> int:
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            return max(0, self.max_requests - len(self.requests))

rate_limiter = RateLimiter(max_requests=RATE_LIMIT)

# ============================================================
# RESPONSE CACHE
# ============================================================

cache = TTLCache(maxsize=100, ttl=3600)  # 100 items, 1 hour TTL
cache_hits = 0
cache_misses = 0

# ============================================================
# LLM CLIENT WITH RETRIES
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def call_llm(prompt: str) -> str:
    """Call LLM with automatic retry on failure."""
    from litellm import completion
    import litellm
    litellm.suppress_debug_info = True
    
    response = completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

# ============================================================
# API MODELS
# ============================================================

class AgentRequest(BaseModel):
    """Request model for agent endpoint."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="The question to ask the AI agent"
    )
    
    class Config:
        json_schema_extra = {
            "example": {"question": "What is machine learning?"}
        }

class AgentResponse(BaseModel):
    """Response model for agent endpoint."""
    answer: str
    request_id: str
    latency_ms: float
    cached: bool

class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    version: str
    model: str
    rate_limit_remaining: int
    cache_hit_rate: str

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="🤖 AI Agent API",
    description="""
## Production-Ready AI Agent API

This API demonstrates production deployment patterns:
- ✅ **Rate Limiting** - 30 requests/minute
- ✅ **Response Caching** - Instant cached responses
- ✅ **Automatic Retries** - Handles temporary failures
- ✅ **Structured Logging** - JSON logs for monitoring

### Endpoints
- `POST /agent` - Ask a question
- `GET /health` - Service health check

### Try it!
Use the **"Try it out"** button below to test the API.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    log("INFO", "AI Agent API started", 
        model=MODEL_NAME, 
        rate_limit=RATE_LIMIT,
        api_key_set=bool(GROQ_API_KEY))

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["System"])
async def root():
    """Welcome endpoint with API info."""
    return {
        "message": "🤖 AI Agent API is running!",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs - Interactive API documentation",
            "agent": "POST /agent - Ask the AI agent",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and statistics.
    """
    total = cache_hits + cache_misses
    hit_rate = (cache_hits / total * 100) if total > 0 else 0
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=MODEL_NAME,
        rate_limit_remaining=rate_limiter.remaining(),
        cache_hit_rate=f"{hit_rate:.1f}%"
    )


@app.post("/agent", response_model=AgentResponse, tags=["Agent"])
async def ask_agent(request: AgentRequest):
    """
    Ask the AI Agent a question.
    
    Features:
    - **Rate Limited**: 30 requests per minute
    - **Cached**: Repeated questions return instantly
    - **Retries**: Automatic retry on temporary failures
    
    Returns the AI's answer with metadata.
    """
    global cache_hits, cache_misses
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    log("INFO", "Request received", 
        request_id=request_id,
        question_length=len(request.question))
    
    # Check rate limit
    if not rate_limiter.allow():
        log("WARNING", "Rate limit exceeded", request_id=request_id)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait and try again."
        )
    
    # Check cache
    cache_key = hashlib.md5(request.question.encode()).hexdigest()
    
    if cache_key in cache:
        cache_hits += 1
        elapsed = time.time() - start_time
        log("INFO", "Cache hit", request_id=request_id, latency_ms=round(elapsed*1000, 2))
        
        return AgentResponse(
            answer=cache[cache_key],
            request_id=request_id,
            latency_ms=round(elapsed * 1000, 2),
            cached=True
        )
    
    # Call LLM
    cache_misses += 1
    
    try:
        answer = call_llm(request.question)
        cache[cache_key] = answer
        
        elapsed = time.time() - start_time
        log("INFO", "Request completed",
            request_id=request_id,
            latency_ms=round(elapsed*1000, 2),
            cached=False)
        
        return AgentResponse(
            answer=answer,
            request_id=request_id,
            latency_ms=round(elapsed * 1000, 2),
            cached=False
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        log("ERROR", "Request failed",
            request_id=request_id,
            error=str(e),
            latency_ms=round(elapsed*1000, 2))
        
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║  🤖 AI Agent API                                     ║
    ║                                                      ║
    ║  Local URL:  http://localhost:{port}                  ║
    ║  API Docs:   http://localhost:{port}/docs             ║
    ║  Health:     http://localhost:{port}/health           ║
    ║                                                      ║
    ║  Press Ctrl+C to stop                                ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
