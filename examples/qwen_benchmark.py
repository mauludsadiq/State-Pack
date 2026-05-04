#!/usr/bin/env python3
"""
State Pack benchmark on Qwen2.5-3B.
Run: PYTHONPATH=. python3 examples/qwen_benchmark.py
"""
import json, time, torch, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from state_pack.store import PacketStore, _sha256
from state_pack.serialize import save_kv_cache, to_dynamic_cache, from_dynamic_cache

MODEL_ID = "Qwen/Qwen2.5-3B"
STORE    = "demo/qwen_store"
STEPS    = 20

print(f"Loading {MODEL_ID}...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model.eval()
print("Model loaded.")

base_text = (
    "You are an autonomous legal research agent. "
    "Track instructions, prior tool results, citations, constraints, and open questions. "
    "Preserve full working memory and continue from prior state.\n\n"
)

deltas = [
    f"Step {i}: Tool observation says contract clause {i} affects indemnity, waiver, notice, or damages.\n"
    for i in range(1, STEPS + 1)
]

ps = PacketStore(store=STORE, model_id=MODEL_ID)

print("Running base pass...")
base_ids = tok(base_text, return_tensors="pt", add_special_tokens=False)
t0 = time.perf_counter()
with torch.no_grad():
    base_out = model(**base_ids, use_cache=True)
base_s = time.perf_counter() - t0
base_tokens = base_ids["input_ids"].shape[1]
base_sha256 = _sha256(base_text)
past = to_dynamic_cache(base_out.past_key_values)
print(f"Base: {base_tokens} tokens in {base_s:.2f}s")

ps.create(base_text, base_out.past_key_values, base_tokens, dtype=torch.float16)
blob_path = list(pathlib.Path(STORE).glob(f"state_packet_{base_sha256}.pt"))
blob_size = blob_path[0].stat().st_size if blob_path else 0
print(f"Blob size: {blob_size/1e6:.1f}MB")

naive_tokens = 0
sp_tokens    = base_tokens
naive_s      = 0.0
sp_s         = base_s
per_step     = []
context      = base_text

print(f"\nRunning {STEPS} steps...")
for i, delta in enumerate(deltas, 1):
    context += delta

    full_ids = tok(context, return_tensors="pt", add_special_tokens=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**full_ids, use_cache=True)
    naive_step_s    = time.perf_counter() - t0
    naive_step_tokens = full_ids["input_ids"].shape[1]

    delta_ids = tok(delta, return_tensors="pt", add_special_tokens=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )
    sp_step_s    = time.perf_counter() - t0
    past         = out.past_key_values
    delta_tokens = delta_ids["input_ids"].shape[1]

    naive_tokens += naive_step_tokens
    sp_tokens    += delta_tokens
    naive_s      += naive_step_s
    sp_s         += sp_step_s

    saved   = naive_tokens - sp_tokens
    savings = round(saved / naive_tokens * 100, 2) if naive_tokens else 0

    per_step.append({
        "step": i,
        "naive_tokens": naive_step_tokens,
        "sp_tokens": delta_tokens,
        "cumulative_naive": naive_tokens,
        "cumulative_sp": sp_tokens,
        "savings_pct": savings,
        "naive_s": round(naive_step_s, 3),
        "sp_s": round(sp_step_s, 3),
    })

    if i % 5 == 0:
        print(f"  step {i:3d}: naive={naive_step_tokens:,}tok  "
              f"sp={delta_tokens}tok  "
              f"savings={savings:.1f}%  "
              f"naive={naive_step_s:.2f}s  sp={sp_step_s:.3f}s")

total_saved = naive_tokens - sp_tokens
savings_pct = round(total_saved / naive_tokens * 100, 2)
speedup     = round(naive_s / sp_s, 3) if sp_s > 0 else 0

result = {
    "model":      MODEL_ID,
    "model_params": "3B",
    "steps":      STEPS,
    "base_tokens": base_tokens,
    "blob_size_mb": round(blob_size / 1e6, 1),
    "naive": {
        "tokens":  naive_tokens,
        "seconds": round(naive_s, 2),
        "avg_tokens_per_step": round(naive_tokens / STEPS, 1),
    },
    "state_pack": {
        "tokens":  sp_tokens,
        "seconds": round(sp_s, 2),
        "avg_tokens_per_step": round(sp_tokens / STEPS, 1),
    },
    "savings": {
        "tokens_saved": total_saved,
        "savings_pct":  savings_pct,
        "speedup":      speedup,
    },
    "per_step": per_step,
}

pathlib.Path("demo").mkdir(exist_ok=True)
pathlib.Path("demo/qwen_benchmark.json").write_text(json.dumps(result, indent=2))

print(f"""
{'='*52}
  Model:         {MODEL_ID}
  Steps:         {STEPS}
{'='*52}
  Naive tokens:  {naive_tokens:,}
  SP tokens:     {sp_tokens:,}
  Tokens saved:  {total_saved:,} ({savings_pct}%)
  Speedup:       {speedup}x
  Blob size:     {blob_size/1e6:.1f}MB
{'='*52}
""")
