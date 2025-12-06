  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
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
#!/usr/bin/env python3
"""
Simple canary traffic generator for canary tests.

Usage:
  python3 scripts/simulate_canary_load.py --url http://aegis-staging.example.com/infer --qps 20 --duration 60 --error-rate 0.05
"""
from __future__ import annotations
import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

def send_request(url: str, payload: dict, timeout: float = 5.0):
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        return r.status_code
    except Exception:
        return 0

def make_payload(bad=False):
    if bad:
        return {"input": "MALFORMED"}
    # default synthetic example for image model
    return {"input": [[[[0.0]] * 1] * 28] * 28}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--qps", type=int, default=10)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--error-rate", type=float, default=0.0)
    args = ap.parse_args()

    end = time.time() + args.duration
    pool = ThreadPoolExecutor(max_workers=100)
    futures = []
    while time.time() < end:
        for _ in range(args.qps):
            bad = random.random() < args.error_rate
            payload = make_payload(bad=bad)
            futures.append(pool.submit(send_request, args.url, payload))
        time.sleep(1)

    total = 0
    ok = 0
    for f in as_completed(futures):
        status = f.result()
        total += 1
        if 200 <= status < 300:
            ok += 1
    print(f"Sent {total} requests, successful: {ok}/{total}")

if __name__ == "__main__":
    main()
scripts/simulate_canary_load.py
