
"""
夭夭 API 单元测试 - 支持流式 + 非流式 + 每token计时
"""
import requests
import time
import sys
import json

BASE = "http://127.0.0.1:11434"

def test_models():
    r = requests.get(f"{BASE}/v1/models")
    r.raise_for_status()
    print("=== /v1/models ===")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

def test_non_stream(messages, max_tokens=60, temperature=0.9, top_p=0.9, label=""):
    print(f"\n=== Test: {label} (non-stream) ===")
    print(f"Input: {messages[-1]['content'][:80]}")
    t0=time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "yaoyao-v0.9",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False
    })
    r.raise_for_status()
    j = r.json()
    text=j['choices'][0]['message']['content']
    elapsed=time.time()-t0
    tps=len(text.split())/elapsed if elapsed>0 else 0
    print(f"Output: {text}")
    print(f"[Time: {elapsed*1000:.0f}ms, ~{tps:.1f} tokens/sec]")
    return j

def test_stream(messages, max_tokens=60, temperature=0.9, top_p=0.9, label=""):
    print(f"\n=== Test: {label} (STREAM with per-token timing) ===")
    print(f"Input: {messages[-1]['content'][:80]}")
    t0=time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "yaoyao-v0.9",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True
    }, stream=True, headers={"Accept":"text/event-stream"})
    r.raise_for_status()
    print("Output: ", end="", flush=True)
    token_count=0
    first_token_time=None
    chunk_times=[]
    last=time.time()
    print()
    print("Token stream:")
    for line in r.iter_lines():
        if not line: continue
        if isinstance(line, bytes): line=line.decode('utf-8',errors='replace')
        if line.startswith("data: ") and line.strip()!="data: [DONE]":
            try:
                data=json.loads(line[6:])
                delta=data.get('choices',[{}])[0].get('delta',{})
                content=delta.get('content','')
                timing=data.get('timing',{})
                step_ms=timing.get('step_ms',0)
                total_ms=timing.get('total_ms',0)
                if content and content.strip():
                    if first_token_time is None:
                        first_token_time=time.time()
                    print(f"  [{token_count:2d}] {content.rstrip():15s} step={step_ms:.1f}ms total={total_ms:.1f}ms")
                    token_count+=1
                    chunk_times.append(time.time()-last)
                    last=time.time()
            except: pass
    elapsed=time.time()-t0
    if first_token_time:
        ttft=(first_token_time-t0)*1000
    else:
        ttft=0
    tps=token_count/elapsed if elapsed>0 else 0
    avg_interval=sum(chunk_times)/len(chunk_times)*1000 if chunk_times else 0
    print(f"\n[Stats: tokens={token_count}, total={elapsed*1000:.0f}ms, TTFT={ttft:.0f}ms, {tps:.1f} tok/s, avg interval={avg_interval:.1f}ms]")
    return token_count, elapsed

if __name__ == "__main__":
    try:
        test_models()
    except Exception as e:
        print(f"Cannot connect to {BASE}: {e}")
        print("Start server first: yaoyao_api.py")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("NON-STREAMING TESTS")
    print("="*60)
    test_non_stream([{"role":"user","content":"Once upon a time"}], label="Story", max_tokens=40)
    test_non_stream([{"role":"user","content":"What is the sky?"}], label="Q&A", max_tokens=20, temperature=0.7)
    
    print("\n" + "="*60)
    print("STREAMING TESTS (with per-token timing)")
    print("="*60)
    test_stream([{"role":"user","content":"Once upon a time"}], label="Story (stream)", max_tokens=20)
    test_stream([{"role":"user","content":"The little girl"}], label="Character (stream)", max_tokens=20)
    test_stream([{"role":"user","content":"Lily and Tom"}], label="Named chars (stream)", max_tokens=20)
    
    print("\n=== All tests passed! ===")
