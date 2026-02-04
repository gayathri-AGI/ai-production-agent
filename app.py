"""
🚀 Production AI Agent API with Beautiful UI
Agentic Security-Focused RAG enabled
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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache

# 🔹 RAG IMPORT
from rag_retriever import retrieve_security_context

# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "groq/llama-3.1-8b-instant")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "30"))

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set!")

# ============================================================
# LOGGING
# ============================================================

def log(level: str, message: str, **kwargs):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "service": "ai-agent-api",
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry))

# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
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
# CACHE
# ============================================================

cache = TTLCache(maxsize=100, ttl=3600)
cache_hits = 0
cache_misses = 0

# ============================================================
# LLM CLIENT
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=60))
def call_llm(prompt: str) -> str:
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
    question: str = Field(..., min_length=1, max_length=5000)

class AgentResponse(BaseModel):
    answer: str
    request_id: str
    latency_ms: float
    cached: bool

# ============================================================
# BEAUTIFUL HTML UI
# ============================================================

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 AI Agent</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg,#667eea,#764ba2);
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
        }
        .box {
            background:white;
            padding:30px;
            border-radius:12px;
            width:500px;
            box-shadow:0 10px 30px rgba(0,0,0,.3);
        }
        textarea {
            width:100%;
            height:100px;
            margin-bottom:15px;
            padding:10px;
            font-size:16px;
        }
        button {
            width:100%;
            padding:12px;
            background:#667eea;
            border:none;
            color:white;
            font-size:16px;
            cursor:pointer;
            border-radius:6px;
        }
        .output {
            margin-top:20px;
            white-space:pre-wrap;
        }
    </style>
</head>
<body>
<div class="box">
    <h2>🤖 AI Agent</h2>
    <textarea id="q">What is machine learning?</textarea>
    <button onclick="ask()">Ask</button>
    <div class="output" id="out"></div>
</div>

<script>
async function ask() {
    const q = document.getElementById("q").value;
    const res = await fetch("/agent", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body:JSON.stringify({ question:q })
    });
    const data = await res.json();
    document.getElementById("out").innerText = data.answer || data.detail;
}
</script>
</body>
</html>
"""

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Agent API",
    description="Production-ready AI Agent with RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    log("INFO", "AI Agent API started", model=MODEL_NAME, rag="enabled")

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_UI

@app.post("/agent", response_model=AgentResponse)
async def ask_agent(request: AgentRequest):
    global cache_hits, cache_misses

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    if not rate_limiter.allow():
        raise HTTPException(429, "Rate limit exceeded")

    cache_key = hashlib.md5(request.question.encode()).hexdigest()

    if cache_key in cache:
        cache_hits += 1
        return AgentResponse(
            answer=cache[cache_key],
            request_id=request_id,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            cached=True,
        )

    cache_misses += 1

    try:
        context = retrieve_security_context(request.question)

        prompt = f"""
You are a security-aware AI agent.

SECURITY RULES:
{context}

USER QUESTION:
{request.question}

Follow the rules strictly.
"""

        answer = call_llm(prompt)
        cache[cache_key] = answer

        return AgentResponse(
            answer=answer,
            request_id=request_id,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            cached=False,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
