"""
state_pack.agent_loop
~~~~~~~~~~~~~~~~~~~~~
Drop-in agent loop that uses StatePack for KV cache reuse.
Replaces the raw HuggingFace loop in examples/agent_loop_benchmark.py.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from .client import StatePack, _sha256


class AgentLoop:
    """
    Runs an N-step agent loop using State Pack for token savings.

    Args:
        model:       HuggingFace CausalLM model (already loaded, eval mode)
        tokenizer:   Matching tokenizer
        store:       Path to State Pack store directory
        model_id:    Model identifier string (e.g. "gpt2")
        merge_every: Merge KV cache into base every N steps (0 = never)
    """

    def __init__(self, model, tokenizer, store="sp_store", model_id="gpt2", merge_every=0):
        self.model = model
        self.tok = tokenizer
        self.sp = StatePack(store=store, model_id=model_id)
        self.merge_every = merge_every
        self.model_id = model_id

    def run(self, base_text: str, deltas: list[str]) -> dict:
        """
        Run base + N delta steps. Returns benchmark results dict.
        """
        model, tok = self.model, self.tok

        # --- Base pass ---
        base_ids = tok(base_text, return_tensors="pt", add_special_tokens=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            base_out = model(**base_ids, use_cache=True)
        base_s = time.perf_counter() - t0

        base_tokens = base_ids["input_ids"].shape[1]
        base_sha256 = _sha256(base_text)
        past = base_out.past_key_values

        # Register in store
        self.sp.create(base_text, past, base_tokens)

        naive_tokens = 0
        sp_tokens = base_tokens
        naive_s = 0.0
        sp_s = base_s
        per_step = []
        current_sha256 = base_sha256
        current_tokens = base_tokens

        for i, delta in enumerate(deltas, start=1):
            # Naive: reprocess full context
            full_text = base_text + "".join(deltas[:i])
            full_ids = tok(full_text, return_tensors="pt", add_special_tokens=False)
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**full_ids, use_cache=True)
            naive_step_s = time.perf_counter() - t0
            naive_step_tokens = full_ids["input_ids"].shape[1]

            # State Pack: delta only
            delta_ids = tok(delta, return_tensors="pt", add_special_tokens=False)
            t0 = time.perf_counter()
            with torch.no_grad():
                delta_out = model(
                    input_ids=delta_ids["input_ids"],
                    past_key_values=past,
                    use_cache=True,
                )
            sp_step_s = time.perf_counter() - t0
            past = delta_out.past_key_values
            delta_tokens = delta_ids["input_ids"].shape[1]

            # Emit infer receipt
            self.sp.infer(current_sha256, delta)

            naive_tokens += naive_step_tokens
            sp_tokens += delta_tokens
            naive_s += naive_step_s
            sp_s += sp_step_s

            # Optional merge
            should_merge = self.merge_every > 0 and i % self.merge_every == 0
            if should_merge:
                new_tokens = current_tokens + delta_tokens
                self.sp.merge(current_sha256, delta, past, new_tokens)
                # Advance current state
                merged_pre = f"{current_sha256}|{_sha256(delta)}"
                current_sha256 = _sha256(merged_pre)
                current_tokens = new_tokens

            saved = naive_tokens - sp_tokens
            per_step.append({
                "step": i,
                "naive_tokens": naive_step_tokens,
                "sp_tokens": delta_tokens,
                "saved": naive_step_tokens - delta_tokens,
                "cumulative_naive": naive_tokens,
                "cumulative_sp": sp_tokens,
                "cumulative_saved": saved,
                "savings_pct": round(saved / naive_tokens * 100, 2) if naive_tokens else 0,
                "naive_s": round(naive_step_s, 4),
                "sp_s": round(sp_step_s, 4),
                "merged": should_merge,
            })

        total_saved = naive_tokens - sp_tokens
        return {
            "model": self.model_id,
            "steps": len(deltas),
            "naive": {"tokens": naive_tokens, "seconds": round(naive_s, 4)},
            "state_pack": {"tokens": sp_tokens, "seconds": round(sp_s, 4)},
            "savings": {
                "tokens_saved": total_saved,
                "savings_pct": round(total_saved / naive_tokens * 100, 2) if naive_tokens else 0,
                "speedup": round(naive_s / sp_s, 3) if sp_s > 0 else 0,
            },
            "per_step": per_step,
        }
