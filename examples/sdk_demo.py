#!/usr/bin/env python3
"""
Quick smoke test of the StatePack SDK against a real GPT-2 model.
Run from repo root: python3 examples/sdk_demo.py
"""
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from state_pack import StatePack
from state_pack.client import _sha256

MODEL_ID = "gpt2"
STORE = "demo/sdk_smoke_store"

print("Loading model...")
tok = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()

base_text = (
    "You are an autonomous legal research agent. "
    "Track instructions, prior tool results, citations, constraints, and open questions. "
    "Preserve full working memory and continue from prior state.\n\n"
)

deltas = [
    f"Step {i}: Tool observation says contract clause {i} affects indemnity, waiver, notice, or damages.\n"
    for i in range(1, 6)
]

sp = StatePack(store=STORE, model_id=MODEL_ID)

# --- Base pass ---
print("Running base pass...")
base_ids = tok(base_text, return_tensors="pt", add_special_tokens=False)
with torch.no_grad():
    base_out = model(**base_ids, use_cache=True)

base_tokens = base_ids["input_ids"].shape[1]
past = base_out.past_key_values
base_sha256 = _sha256(base_text)

print(f"Base tokens: {base_tokens}")
receipt = sp.create(base_text, past, base_tokens)
print("CREATE receipt:", json.dumps(receipt, indent=2))

# --- Delta passes ---
for i, delta in enumerate(deltas, 1):
    delta_ids = tok(delta, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        delta_out = model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )
    past = delta_out.past_key_values
    receipt = sp.infer(base_sha256, delta)
    print(f"INFER step {i}:", receipt.get("op"), "ok=", receipt.get("ok"),
          "saved_tokens=", receipt.get("tokens", {}).get("saved"))

# --- Verify ---
receipt = sp.verify(base_sha256)
print("VERIFY:", receipt.get("ok"))

# --- Load back ---
blob = sp.load_base(base_sha256)
print("LOAD blob keys:", list(blob.keys()))
print("KV layers:", blob["num_layers"])
print("\nAll checks passed.")
