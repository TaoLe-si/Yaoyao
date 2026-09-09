import json, urllib.request, sys, time
req = urllib.request.Request(
    "http://127.0.0.1:11434/v1/chat/completions",
    data=json.dumps({
        "model": "yaoyao-v0.9",
        "messages": [{"role": "user", "content": "Once upon a time"}],
        "max_tokens": 200,
        "temperature": 0.9,
        "top_p": 0.9,
        "stream": True,
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
tokens = []
step_ms = []
total_ms = []
received = []
started = time.perf_counter()
with urllib.request.urlopen(req, timeout=300) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip()
        received.append(line)
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            obj = json.loads(body)
        except Exception:
            print("RAW_BODY:", body, flush=True); continue
        choice = obj.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")
        timing = obj.get("timing") or choice.get("timing")
        if content:
            tokens.append(content)
            if timing:
                sm = float(timing.get("step_ms", 0))
                tm = float(timing.get("total_ms", 0))
                step_ms.append(sm)
                total_ms.append(tm)
                sys.stdout.write(f"[step={sm:.3f}ms total={tm:.3f}ms] {content}")
                sys.stdout.flush()
print("\n---SUMMARY---")
print("lines_received:", len(received))
print("text:", "".join(tokens))
if step_ms:
    avg = sum(step_ms)/len(step_ms)
    print(f"tokens={len(step_ms)} avg_step_ms={avg:.3f} max_step_ms={max(step_ms):.3f} server_total_ms={total_ms[-1]:.3f} tokens_per_sec={len(step_ms)*1000.0/max(total_ms[-1],1e-3):.2f}")
print(f"http_wall_ms={(time.perf_counter()-started)*1000:.3f}")
print("first_sse_lines:")
for line in received[:8]:
    print(repr(line))