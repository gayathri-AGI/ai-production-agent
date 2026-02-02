"""
🚀 Production AI Agent API with Beautiful UI
   Ready for deployment to Render / Railway
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

# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
MODEL_NAME = os.environ.get('MODEL_NAME', 'groq/llama-3.1-8b-instant')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '512'))
RATE_LIMIT = int(os.environ.get('RATE_LIMIT', '30'))

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
        **kwargs
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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 { text-align: center; color: #333; margin-bottom: 10px; font-size: 2.5rem; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
            margin-bottom: 20px;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102,126,234,0.4); }
        button:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        
        .response-box {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            display: none;
        }
        .response-box.show { display: block; }
        .response-label { font-weight: bold; color: #667eea; margin-bottom: 10px; display: block; }
        .response-text { color: #333; line-height: 1.6; white-space: pre-wrap; }
        
        .stats {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .stat { font-size: 12px; color: #888; }
        .stat-value { font-weight: bold; color: #667eea; }
        .cached-badge { background: #4CAF50; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
        .error { color: #e74c3c; background: #fdeaea; padding: 10px; border-radius: 8px; }
        .loading { text-align: center; color: #667eea; }
        
        .features {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        .feature { font-size: 12px; color: #888; background: #f0f0f0; padding: 5px 10px; border-radius: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Agent</h1>
        <p class="subtitle">Ask me anything!</p>
        
        <div class="features">
            <span class="feature">⚡ Fast</span>
            <span class="feature">🔄 Cached</span>
            <span class="feature">🛡️ Rate Limited</span>
        </div>
        
        <textarea id="question" placeholder="Type your question here..." rows="3">What is machine learning?</textarea>
        <button id="askBtn" onclick="askAgent()">Ask AI Agent 🚀</button>
        
        <div id="responseBox" class="response-box">
            <span class="response-label">AI Response:</span>
            <div id="responseText" class="response-text"></div>
            <div id="stats" class="stats"></div>
        </div>
    </div>

    <script>
        async function askAgent() {
            const question = document.getElementById('question').value.trim();
            const btn = document.getElementById('askBtn');
            const responseBox = document.getElementById('responseBox');
            const responseText = document.getElementById('responseText');
            const stats = document.getElementById('stats');
            
            if (!question) { alert('Please enter a question!'); return; }
            
            btn.disabled = true;
            btn.textContent = '⏳ Thinking...';
            responseBox.classList.add('show');
            responseText.innerHTML = '<div class="loading">🔄 Getting response...</div>';
            stats.innerHTML = '';
            
            try {
                const startTime = Date.now();
                const response = await fetch('/agent', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                const clientTime = Date.now() - startTime;
                
                if (!response.ok) {
                    if (response.status === 429) throw new Error('⚠️ Rate limit exceeded! Wait a moment.');
                    throw new Error('Error: ' + response.status);
                }
                
                const data = await response.json();
                responseText.textContent = data.answer;
                responseText.className = 'response-text';
                
                const cachedBadge = data.cached ? '<span class="cached-badge">⚡ CACHED</span>' : '';
                stats.innerHTML = `
                    <span class="stat">ID: <span class="stat-value">${data.request_id}</span></span>
                    <span class="stat">Server: <span class="stat-value">${data.latency_ms.toFixed(0)}ms</span></span>
                    <span class="stat">Total: <span class="stat-value">${clientTime}ms</span></span>
                    ${cachedBadge}
                `;
            } catch (error) {
                responseText.textContent = error.message;
                responseText.className = 'response-text error';
            }
            
            btn.disabled = false;
            btn.textContent = 'Ask AI Agent 🚀';
        }
        
        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askAgent(); }
        });
    </script>
</body>
</html>
"""

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="🤖 AI Agent API",
    description="Production-ready AI Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    log("INFO", "AI Agent API started", model=MODEL_NAME)

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Beautiful UI for the AI Agent."""
    return HTML_UI


@app.get("/health")
async def health():
    total = cache_hits + cache_misses
    hit_rate = (cache_hits / total * 100) if total > 0 else 0
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "rate_limit_remaining": rate_limiter.remaining(),
        "cache_hit_rate": f"{hit_rate:.1f}%"
    }


@app.post("/agent", response_model=AgentResponse)
async def ask_agent(request: AgentRequest):
    global cache_hits, cache_misses
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    log("INFO", "Request received", request_id=request_id)
    
    if not rate_limiter.allow():
        log("WARNING", "Rate limit exceeded", request_id=request_id)
        raise HTTPException(429, "Rate limit exceeded")
    
    cache_key = hashlib.md5(request.question.encode()).hexdigest()
    
    if cache_key in cache:
        cache_hits += 1
        elapsed = time.time() - start_time
        log("INFO", "Cache hit", request_id=request_id)
        return AgentResponse(
            answer=cache[cache_key],
            request_id=request_id,
            latency_ms=round(elapsed * 1000, 2),
            cached=True
        )
    
    cache_misses += 1
    
    try:
        answer = call_llm(request.question)
        cache[cache_key] = answer
        elapsed = time.time() - start_time
        log("INFO", "Request completed", request_id=request_id, latency_ms=round(elapsed*1000, 2))
        
        return AgentResponse(
            answer=answer,
            request_id=request_id,
            latency_ms=round(elapsed * 1000, 2),
            cached=False
        )
    except Exception as e:
        log("ERROR", "Request failed", request_id=request_id, error=str(e))
        raise HTTPException(500, f"Agent error: {str(e)}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🤖 AI Agent running at http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
