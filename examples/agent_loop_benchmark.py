#!/usr/bin/env python3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import json

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
        model(**x, use_cache=True)
    return x["input_ids"].shape[1]

def run_delta(delta, past):
    x = tok(delta, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(input_ids=x["input_ids"], past_key_values=past, use_cache=True)
    return out.past_key_values, x["input_ids"].shape[1]

context = base
naive_tokens = 0
naive_seconds = 0.0

for delta in deltas:
    context += delta
    t0 = time.perf_counter()
    n = run_full(context)
    naive_seconds += time.perf_counter() - t0
    naive_tokens += n

base_x = tok(base, return_tensors="pt")
with torch.no_grad():
    base_out = model(**base_x, use_cache=True)

statepack_tokens = base_x["input_ids"].shape[1]
statepack_seconds = 0.0
past = base_out.past_key_values

for delta in deltas:
    t0 = time.perf_counter()
    past, n = run_delta(delta, past)
    statepack_seconds += time.perf_counter() - t0
    statepack_tokens += n

saved_tokens = naive_tokens - statepack_tokens
savings_percent = (saved_tokens / naive_tokens) * 100.0
speedup = naive_seconds / statepack_seconds if statepack_seconds > 0 else float("inf")

print(json.dumps({
    "model": MODEL_ID,
    "steps": STEPS,
    "naive": {
        "tokens_processed": naive_tokens,
        "seconds": naive_seconds
    },
    "state_pack": {
        "tokens_processed": statepack_tokens,
        "seconds": statepack_seconds
    },
    "savings": {
        "tokens_saved": saved_tokens,
        "savings_percent": savings_percent,
        "speedup": speedup
    }
}, indent=2))
