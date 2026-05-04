#!/usr/bin/env python3
"""
Benchmark the State Pack HTTP API over 40 steps.
Server must be running: PYTHONPATH=. python3 -m state_pack.server

Run: python3 examples/api_benchmark.py
"""
import json
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
STEPS = 40

BASE_TEXT = (
    "You are an autonomous legal research agent. "
    "Track instructions, prior tool results, citations, constraints, and open questions. "
    "Preserve full working memory and continue from prior state.\n\n"
)

DELTAS = [
    f"Step {i}: Tool observation says contract clause {i} affects indemnity, waiver, notice, or damages.\n"
    for i in range(1, STEPS + 1)
]


def post(path, payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


# --- Create base packet ---
print("Creating base packet...")
t0 = time.perf_counter()
resp = post("/packets", {"base_text": BASE_TEXT})
create_s = time.perf_counter() - t0
sha = resp["base_sha256"]
base_tokens = resp["base_tokens"]
print(f"  base_sha256: {sha[:16]}...  base_tokens: {base_tokens}  ({create_s:.3f}s)")

# --- Infer steps ---
naive_tokens  = 0
sp_tokens     = base_tokens
total_infer_s = 0.0
per_step      = []

for i, delta in enumerate(DELTAS, 1):
    naive_tokens_this_step = base_tokens + sum(
        len(d.split()) for d in DELTAS[:i]  # rough estimate for naive
    )

    t0 = time.perf_counter()
    r  = post("/infer", {"base_sha256": sha, "delta_text": delta})
    step_s = time.perf_counter() - t0
    total_infer_s += step_s

    delta_tokens = r["tokens"]["delta"]
    saved_tokens = r["tokens"]["saved"]
    sp_tokens   += delta_tokens
    naive_tokens += r["tokens"]["base"] + delta_tokens

    per_step.append({
        "step":         i,
        "delta_tokens": delta_tokens,
        "saved":        saved_tokens,
        "savings_pct":  round(r["tokens"]["savings_percent"], 1),
        "latency_ms":   round(step_s * 1000, 1),
        "next_token":   r["next_token"]["text"].replace("\n", "\\n"),
    })

    if i % 10 == 0:
        print(f"  step {i:3d}: {delta_tokens:3d} delta tokens  "
              f"{saved_tokens:3d} saved  "
              f"{step_s*1000:.0f}ms  "
              f"next='{r['next_token']['text'].replace(chr(10),'↵')}'")

# --- Summary ---
total_saved = naive_tokens - sp_tokens
savings_pct = round(total_saved / naive_tokens * 100, 2) if naive_tokens else 0

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  State Pack API Benchmark — {STEPS} steps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  naive tokens:      {naive_tokens:,}
  state pack tokens: {sp_tokens:,}
  tokens saved:      {total_saved:,}
  savings:           {savings_pct}%
  avg latency/step:  {total_infer_s/STEPS*1000:.1f}ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
