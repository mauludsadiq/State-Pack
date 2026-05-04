#!/usr/bin/env python3
"""
Full agent loop benchmark using the StatePack SDK.
Compares naive (reprocess full context) vs State Pack (delta only).
Run: PYTHONPATH=. python3 examples/sdk_benchmark.py
"""
import json
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from state_pack.agent_loop import AgentLoop

MODEL_ID  = "gpt2"
STORE     = "demo/sdk_benchmark_store"
STEPS     = 40
MERGE_EVERY = 10  # merge KV cache back into base every 10 steps

print("Loading model...")
tok   = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()

base_text = (
    "You are an autonomous legal research agent. "
    "Track instructions, prior tool results, citations, constraints, and open questions. "
    "Preserve full working memory and continue from prior state.\n\n"
)

deltas = [
    f"Step {i}: Tool observation says contract clause {i} affects indemnity, waiver, notice, or damages.\n"
    for i in range(1, STEPS + 1)
]

loop = AgentLoop(model, tok, store=STORE, model_id=MODEL_ID, merge_every=MERGE_EVERY)
print(f"Running {STEPS}-step agent loop (merge every {MERGE_EVERY} steps)...")
results = loop.run(base_text, deltas)

print(json.dumps({
    "model":       results["model"],
    "steps":       results["steps"],
    "naive":       results["naive"],
    "state_pack":  results["state_pack"],
    "savings":     results["savings"],
}, indent=2))

s = results["savings"]
print(f"\n✓ {s['savings_pct']:.1f}% token savings  |  {s['speedup']}x speedup  |  {s['tokens_saved']} tokens saved")
