# -*- coding: utf-8 -*-
"""Simple cross-process GPU lease for D:\Model\final.
Only one GPU-heavy job at a time. Usage:
  python gpu_guard.py acquire --owner deep-vq --min-free-mb 2000
  python gpu_guard.py release
  python gpu_guard.py status
"""
import os, sys, time, json, subprocess, argparse, contextlib
BASE = r"D:\Model\final"
LOCK = os.path.join(BASE, ".gpu_lock")
INFO = os.path.join(LOCK, "owner.json")

def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip().splitlines()[0].strip()
        return int(float(out))
    except Exception:
        return -1

def read_owner():
    try:
        with open(INFO, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def acquire(owner, min_free_mb=2000, timeout=3600, poll=5):
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            with open(INFO, "w", encoding="utf-8") as f:
                json.dump({"owner": owner, "pid": os.getpid(), "start": time.time()}, f)
            free = gpu_free_mb()
            if free < min_free_mb:
                # release and wait
                release()
                print(f"[gpu_guard] acquired but free={free}MB < {min_free_mb}MB; waiting", flush=True)
            else:
                print(f"[gpu_guard] ACQUIRED owner={owner} pid={os.getpid()} free={free}MB", flush=True)
                return True
        except FileExistsError:
            pass
        if time.time() - t0 > timeout:
            print(f"[gpu_guard] TIMEOUT owner={owner} owner_info={read_owner()} free={gpu_free_mb()}MB", flush=True)
            return False
        time.sleep(poll)

def release():
    try:
        if os.path.isdir(LOCK):
            for name in os.listdir(LOCK):
                try: os.remove(os.path.join(LOCK, name))
                except Exception: pass
            os.rmdir(LOCK)
            print("[gpu_guard] RELEASED", flush=True)
            return True
    except Exception as e:
        print(f"[gpu_guard] release error: {e}", flush=True)
    return False

def status():
    print(json.dumps({"lock_exists": os.path.isdir(LOCK), "owner": read_owner(), "free_mb": gpu_free_mb()}, ensure_ascii=False), flush=True)

@contextlib.contextmanager
def lease(owner, min_free_mb=2000, timeout=3600):
    ok = acquire(owner, min_free_mb, timeout)
    if not ok:
        raise RuntimeError("GPU lease timeout")
    try:
        yield
    finally:
        release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acquire", "release", "status"])
    ap.add_argument("--owner", default="unknown")
    ap.add_argument("--min-free-mb", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    if a.cmd == "acquire":
        sys.exit(0 if acquire(a.owner, a.min_free_mb, a.timeout) else 1)
    elif a.cmd == "release":
        release()
    else:
        status()
