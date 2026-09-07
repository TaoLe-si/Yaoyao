"""
夭夭 (Yaoyao) OpenAI-compatible API server
Wraps yaoyao_gen.exe to provide /v1/chat/completions endpoint.
"""
import subprocess
import time
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

YAOYAO_EXE = r"D:\TaoVm\yaoyao_gen.exe"
MODEL_PATH = r"D:\TaoVm\yaoyao_v09_best_loss311.bin"

app = FastAPI(title="Yaoyao OpenAI-compatible API", version="0.9")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "yaoyao-v0.9"
    messages: List[Message]
    max_tokens: int = 60
    temperature: float = 0.9
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def build_prompt(messages: List[Message]) -> str:
    """Convert OpenAI chat messages to a single prompt for Yaoyao."""
    parts = []
    has_system = any(m.role == "system" for m in messages)
    for m in messages:
        role = m.role
        content = m.content.strip()
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(content)
    prompt = "\n\n".join(parts)
    return prompt

def call_yaoyao(prompt: str, max_tokens: int, temperature: float, top_p: float) -> tuple[str, float]:
    """Run yaoyao_gen.exe and parse response. Returns (text, elapsed_sec)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [YAOYAO_EXE, MODEL_PATH, prompt, str(max_tokens), str(temperature), str(top_p)],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Yaoyao timeout")
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Yaoyao error: {proc.stderr[:500]}")
    # Parse RESPONSE: line
    response = ""
    for line in proc.stdout.split("\n"):
        if line.startswith("RESPONSE:"):
            response = "\n".join(proc.stdout.split("\n")[proc.stdout.split("\n").index(line)+1:]).strip()
            break
    return response, elapsed

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "yaoyao-v0.9",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "yaoyao-local",
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")
    prompt = build_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    
    response_text, elapsed = call_yaoyao(prompt, req.max_tokens, req.temperature, req.top_p)
    
    # Truncate at stop sequences
    if req.stop:
        for stop in req.stop:
            idx = response_text.find(stop)
            if idx >= 0:
                response_text = response_text[:idx]
    
    return {
        "id": f"yaoyao-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
    }

@app.get("/")
async def root():
    return {
        "name": "Yaoyao API",
        "version": "0.9",
        "endpoints": ["/v1/chat/completions", "/v1/models"],
        "model": "yaoyao-v0.9 (D=128, NL=2, SEQ=64, V=1024 word-level)",
        "loss": 3.11
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Yaoyao API on http://127.0.0.1:11434")
    print(f"Model: {MODEL_PATH}")
    uvicorn.run(app, host="127.0.0.1", port=11434)
