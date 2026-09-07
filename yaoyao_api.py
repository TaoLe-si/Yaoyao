"""
夭夭 (Yaoyao) OpenAI-compatible API server with streaming support.
"""
import subprocess
import time
import json
import asyncio
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

YAOYAO_EXE = "D:/TaoVm/yaoyao_gen.exe"
MODEL_PATH = "D:/TaoVm/yaoyao_v09_best_loss311.bin"

app = FastAPI(title="Yaoyao OpenAI-compatible API", version="0.9.5")

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

def build_prompt(messages: List[Message]) -> str:
    parts = []
    for m in messages:
        parts.append(m.content.strip())
    return "\n\n".join(parts)

def gen_non_stream(prompt: str, max_tokens: int, T: float, top_p: float) -> tuple[str, float, int]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [YAOYAO_EXE, MODEL_PATH, prompt, str(max_tokens), str(T), str(top_p)],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Yaoyao timeout")
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Yaoyao error: {proc.stderr[:500]}")
    response = ""
    if "RESPONSE:" in proc.stdout:
        idx = proc.stdout.index("RESPONSE:")
        response = proc.stdout[idx + len("RESPONSE:"):].strip()
    return response, elapsed, len(response.split())

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
    prompt = build_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    
    if req.stream:
        # SSE streaming with asyncio subprocess
        async def event_generator():
            chat_id = f"yaoyao-{int(time.time()*1000)}"
            created = int(time.time())
            yield f"data: " + json.dumps({"id":chat_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]}) + "\n\n"
            try:
                proc = await asyncio.create_subprocess_exec(
                    YAOYAO_EXE, MODEL_PATH, "stream", prompt, str(req.max_tokens), str(req.temperature), str(req.top_p),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                while True:
                    line_bytes = await proc.stdout.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode('utf-8', errors='replace').strip()
                    if line.startswith("TOKEN "):
                        parts = line.split(" ", 5)
                        # Format: TOKEN step tok_id step_ms total_ms word
                        if len(parts) >= 6:
                            word = parts[5] + " "
                            total_ms = float(parts[4].rstrip("ms"))
                            step_ms = float(parts[3].rstrip("ms"))
                            yield f"data: " + json.dumps({"id":chat_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{"content":word},"finish_reason":None}],"timing":{"step_ms":step_ms,"total_ms":total_ms}}) + "\n\n"
                    elif line == "DONE":
                        break
                try: await asyncio.wait_for(proc.wait(), timeout=2)
                except: proc.kill()
            except Exception as e:
                yield f"data: " + json.dumps({"error":str(e)}) + "\n\n"
            yield f"data: " + json.dumps({"id":chat_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}) + "\n\n"
            yield "data: [DONE]\\n\\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    # Non-streaming
    response_text, elapsed, completion_tokens = gen_non_stream(prompt, req.max_tokens, req.temperature, req.top_p)
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
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt.split()) + completion_tokens
        }
    }

@app.get("/")
async def root():
    return {
        "name": "Yaoyao API",
        "version": "0.9.5",
        "endpoints": ["/v1/chat/completions", "/v1/models"],
        "features": ["streaming", "OpenAI-compatible"],
        "model": "yaoyao-v0.9 (D=128, NL=2, SEQ=64, V=1024)",
        "loss": 3.11
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Yaoyao API on http://127.0.0.1:11434", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=11434)
