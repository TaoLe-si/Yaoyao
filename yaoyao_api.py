"""
夭夭 (Yaoyao) OpenAI-compatible API server.
Uses persistent C++ server (model loaded once) for low-latency generation.
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

YAOYAO_EXE = "D:/TaoVm/tao_d256_api_state.exe"
MODEL_PATH = "D:/TaoVm/tao_fixed_step50002.bin tao_context_supported.tcg tinystories_train.txt"

app = FastAPI(title="Yaoyao OpenAI-compatible API", version="0.9.6")

# Persistent C++ server
_server_proc = None
_server_ready = asyncio.Event()

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
    return "\\n\\n".join(m.content.strip() for m in messages)

async def start_server():
    global _server_proc
    if _server_proc is not None and _server_proc.returncode is None:
        return
    print(f"Starting persistent Yaoyao server: {YAOYAO_EXE}", flush=True)
    _server_proc = await asyncio.create_subprocess_exec(
        YAOYAO_EXE, "--server",
        "D:/TaoVm/tao_fixed_step50002.bin",
        "D:/TaoVm/tao_context_supported.tcg",
        "D:/TaoVm/tao_coef_rts1_128.reader",
        "D:/TaoVm/tinystories_train.txt",
        "D:/TaoVm/experiments/d256_nibble64_baseline/train_tokens.bin",
        "D:/TaoVm/tao_state_supported_v2_full.tds",
        "D:/TaoVm/tao_alternating_step47002.bin",
        "D:/TaoVm/experiments/d256_nibble64_baseline/multislice/slice_0.bin",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    # Wait for ready signal
    while True:
        line = await _server_proc.stdout.readline()
        if not line:
            break
        text = line.decode('utf-8', errors='replace').strip()
        if "Server ready" in text:
            print(f"Server ready: {text}", flush=True)
            _server_ready.set()
            break

@app.on_event("startup")
async def on_startup():
    await start_server()

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
    await _server_ready.wait()
    prompt = build_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    
    if req.stream:
        async def event_generator():
            chat_id = f"yaoyao-{int(time.time()*1000)}"
            created = int(time.time())
            yield f"data: " + json.dumps({"id":chat_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]}) + "\n\n"
            
            # Send request to persistent server
            req_line = f"{prompt}|1|{req.max_tokens}|{req.temperature}|{req.top_p}\n"
            _server_proc.stdin.write(req_line.encode('utf-8'))
            await _server_proc.stdin.drain()
            
            # Read tokens
            while True:
                line_bytes = await _server_proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if line.startswith("TOKEN "):
                    parts = line.split(" ", 4)
                    if len(parts) >= 5:
                        word = parts[4].replace("\x01", " ") + " "
                        step_ms = float(parts[2])
                        total_ms = float(parts[3])
                        yield f"data: " + json.dumps({"id":chat_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{"content":word},"finish_reason":None}],"timing":{"step_ms":step_ms,"total_ms":total_ms}}) + "\n\n"
                elif line == "DONE":
                    break
                elif line.startswith("ERROR"):
                    break
            
            yield f"data: " + json.dumps({"id":chat_id,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    # Non-streaming
    chat_id = f"yaoyao-{int(time.time()*1000)}"
    created = int(time.time())
    req_line = f"{prompt}|0|{req.max_tokens}|{req.temperature}|{req.top_p}\n"
    _server_proc.stdin.write(req_line.encode('utf-8'))
    await _server_proc.stdin.drain()
    
    response_text = ""
    while True:
        line_bytes = await _server_proc.stdout.readline()
        if not line_bytes:
            break
        line = line_bytes.decode('utf-8', errors='replace').strip()
        if line.startswith("TOKEN "):
            parts = line.split(" ", 4)
            if len(parts) >= 5:
                response_text += parts[4].replace("\x01", " ") + " "
        elif line == "DONE":
            break
        elif line.startswith("ERROR"):
            raise HTTPException(status_code=500, detail=line)
    
    if req.stop:
        for stop in req.stop:
            idx = response_text.find(stop)
            if idx >= 0:
                response_text = response_text[:idx]
    
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text.strip()},
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
        "version": "0.9.6",
        "endpoints": ["/v1/chat/completions", "/v1/models"],
        "features": ["persistent server", "streaming", "OpenAI-compatible"],
        "model": "yaoyao-v0.9 (D=128, NL=2, SEQ=64, V=1024)",
        "loss": 3.11
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Yaoyao API on http://127.0.0.1:11434", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=11434)
