 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
#!/usr/bin/env python3
"""
Simple canary traffic generator.

- Sends requests to a model endpoint to simulate traffic.
- Optionally introduce a fraction of error requests (bad payloads) to simulate a bad canary.
- Useful for testing rollout alert/rollback rules.

Usage:
  python3 scripts/simulate_canary_load.py --url http://aegis-staging.example.com/infer --qps 50 --duration 60 --error-rate 0.05
"""
from __future__ import annotations
import argparse
import httpx
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request(url: str, payload: dict, timeout: float = 5.0):
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--qps", type=int, default=10)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--error-rate", type=float, default=0.0)
    args = ap.parse_args()

    def make_payload(bad=False):
        if bad:
            return {"input": "MALFORMED"}  # deliberately cause error in model
        # example happy path payload for image model: base64 or small synthetic
        return {"input": [[[[0.0]*1]*28]*28]}  # adjust to model input spec

    end = time.time() + args.duration
    pool = ThreadPoolExecutor(max_workers=100)
    futures = []
    sent = 0
    while time.time() < end:
        for _ in range(args.qps):
            bad = random.random() < args.error_rate
            payload = make_payload(bad=bad)
            futures.append(pool.submit(send_request, args.url, payload))
            sent += 1
        # sleep 1 second per QPS loop
        time.sleep(1)

    ok = 0
    total = 0
    for f in as_completed(futures):
        status, body = f.result()
        total += 1
        if status >= 200 and status < 300:
            ok += 1
    print(f"Sent {sent} requests, successful: {ok}/{total}")

if __name__ == "__main__":
    main()
scripts/simulate_canary_load.py
