#!/usr/bin/env python3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import torch
import time
import json
import hashlib
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--input-cost-per-m", type=float, default=0.0)
parser.add_argument("--out")
args = parser.parse_args()

MODEL_ID = "gpt2"
STEPS = 40

tok = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()

base = (
    "You are an autonomous legal research agent. "
    "You must track instructions, prior tool results, citations, constraints, and unresolved questions. "
    "Always preserve the full working memory and continue from prior state.\n\n"
)

deltas = [
    f"Step {i}: Tool observation says contract clause {i} affects indemnity, waiver, notice, or damages.\n"
    for i in range(1, STEPS + 1)
]

def run_full(text):
    x = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**x, use_cache=True)
    return x["input_ids"].shape[1], out.logits[:, -1, :]

def run_delta(delta, past):
    x = tok(delta, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(input_ids=x["input_ids"], past_key_values=past, use_cache=True)
    return out.past_key_values, x["input_ids"].shape[1], out.logits[:, -1, :]

def sample_token(logits):
    token_id = int(torch.argmax(logits, dim=-1)[0].item())
    return token_id, tok.decode([token_id])

context = base
naive_tokens = 0
naive_seconds = 0.0
steps = []

base_x = tok(base, return_tensors="pt")
with torch.no_grad():
    base_out = model(**base_x, use_cache=True)

statepack_tokens = base_x["input_ids"].shape[1]
statepack_seconds = 0.0
past = base_out.past_key_values
final_sample = None

for i, delta in enumerate(deltas, start=1):
    context += delta

    t0 = time.perf_counter()
    full_tokens_this_step, full_logits = run_full(context)
    full_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    past, delta_tokens_this_step, delta_logits = run_delta(delta, past)
    delta_s = time.perf_counter() - t1

    naive_tokens += full_tokens_this_step
    naive_seconds += full_s
    statepack_tokens += delta_tokens_this_step
    statepack_seconds += delta_s

    cumulative_saved = naive_tokens - statepack_tokens
    cumulative_savings_percent = (cumulative_saved / naive_tokens) * 100.0 if naive_tokens else 0.0

    token_id, text = sample_token(delta_logits)
    final_sample = {"token_id": token_id, "text": text}

    steps.append({
        "step": i,
        "naive_tokens": full_tokens_this_step,
        "state_pack_tokens": delta_tokens_this_step,
        "tokens_saved_this_step": full_tokens_this_step - delta_tokens_this_step,
        "cumulative_naive_tokens": naive_tokens,
        "cumulative_state_pack_tokens": statepack_tokens,
        "cumulative_tokens_saved": cumulative_saved,
        "cumulative_savings_percent": cumulative_savings_percent,
        "naive_seconds": full_s,
        "state_pack_seconds": delta_s
    })

saved_tokens = naive_tokens - statepack_tokens
savings_percent = (saved_tokens / naive_tokens) * 100.0
speedup = naive_seconds / statepack_seconds if statepack_seconds > 0 else float("inf")
estimated_usd_saved = (saved_tokens / 1_000_000.0) * args.input_cost_per_m

result = {
    "op": "benchmark",
    "model": MODEL_ID,
    "steps": STEPS,
    "naive": {
        "tokens_processed": naive_tokens,
        "seconds": naive_seconds,
        "avg_tokens_per_step": naive_tokens / STEPS
    },
    "state_pack": {
        "tokens_processed": statepack_tokens,
        "seconds": statepack_seconds,
        "avg_tokens_per_step": statepack_tokens / STEPS
    },
    "savings": {
        "tokens_saved": saved_tokens,
        "savings_percent": savings_percent,
        "speedup": speedup,
        "estimated_usd_saved": estimated_usd_saved,
        "input_cost_per_m": args.input_cost_per_m
    },
    "final_generated_output_sample": final_sample,
    "per_step": steps,
    "metadata": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
}

canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
result["metadata"]["receipt_id"] = "sha256:" + hashlib.sha256(canonical).hexdigest()

if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
