from __future__ import annotations

"""
state_pack.llm
~~~~~~~~~~~~~~
Drop-in LLM wrapper that transparently applies State Pack KV caching
to any HuggingFace CausalLM model.

Works standalone or as a LangChain-compatible LLM.

Standalone usage:
    from state_pack.llm import StatePackLLM

    llm = StatePackLLM.from_pretrained("gpt2", store="my_store")
    llm.set_base("You are a research agent...\n\n")

    # Each call only processes new tokens
    out = llm("Step 1: observe clause A.")
    out = llm("Step 2: observe clause B.")
    print(llm.stats())

LangChain usage (if langchain installed):
    from langchain.chains import ConversationChain
    chain = ConversationChain(llm=llm)
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .store import PacketStore, _sha256
from .serialize import to_dynamic_cache, from_dynamic_cache


class StatePackLLM:
    """
    HuggingFace CausalLM with transparent State Pack KV caching.

    After set_base() is called, each __call__ processes only the new
    delta tokens against the cached base state.

    Optionally wraps as a LangChain BaseLLM if langchain is installed.
    """

    def __init__(
        self,
        model,
        tokenizer,
        store: str | Path = "sp_store",
        model_id: str = "gpt2",
        max_new_tokens: int = 32,
        merge_every: int = 0,
        dtype: torch.dtype = torch.float16,
    ):
        self.model          = model
        self.tok            = tokenizer
        self.ps             = PacketStore(store=store, model_id=model_id)
        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens
        self.merge_every    = merge_every
        self.dtype          = dtype

        self._base_sha256:   Optional[str]          = None
        self._base_tokens:   int                    = 0
        self._past:          Optional[DynamicCache] = None
        self._step:          int                    = 0
        self._current_sha256: Optional[str]         = None

        # Stats
        self._naive_tokens: int   = 0
        self._sp_tokens:    int   = 0
        self._total_s:      float = 0.0
        self._naive_s:      float = 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        store: str | Path = "sp_store",
        max_new_tokens: int = 32,
        merge_every: int = 0,
        dtype: torch.dtype = torch.float16,
        **model_kwargs,
    ) -> "StatePackLLM":
        tok   = AutoTokenizer.from_pretrained(model_id, **model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        return cls(model, tok, store=store, model_id=model_id,
                   max_new_tokens=max_new_tokens, merge_every=merge_every, dtype=dtype)

    def set_base(self, base_text: str) -> dict:
        """
        Run base_text through the model, cache KV state.
        Must be called before __call__.
        Returns create receipt.
        """
        ids = self.tok(base_text, return_tensors="pt", add_special_tokens=False)
        t0  = time.perf_counter()
        with torch.no_grad():
            out = self.model(**ids, use_cache=True)
        base_s = time.perf_counter() - t0

        base_tokens       = ids["input_ids"].shape[1]
        self._base_sha256 = _sha256(base_text)
        self._base_tokens = base_tokens
        self._past        = to_dynamic_cache(
            tuple(tuple(t.to(self.dtype) for t in l) for l in out.past_key_values)
        )
        self._current_sha256 = self._base_sha256
        self._step           = 0
        self._sp_tokens      = base_tokens
        self._total_s        = base_s

        receipt = self.ps.create(base_text, out.past_key_values, base_tokens, dtype=self.dtype)
        return receipt

    def __call__(self, delta_text: str, generate: bool = False) -> str:
        """
        Process delta_text against cached base state.
        Returns decoded output token(s).
        """
        if self._past is None:
            raise RuntimeError("Call set_base() before invoking the LLM.")

        delta_ids = self.tok(delta_text, return_tensors="pt", add_special_tokens=False)
        delta_tokens = delta_ids["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            if generate:
                out_ids = self.model.generate(
                    delta_ids["input_ids"],
                    past_key_values=self._past,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                )
                result_text = self.tok.decode(
                    out_ids[0][delta_ids["input_ids"].shape[1]:], skip_special_tokens=True
                )
                # Refresh past from a fresh forward pass
                fwd = self.model(input_ids=delta_ids["input_ids"],
                                 past_key_values=self._past, use_cache=True)
                self._past = fwd.past_key_values
            else:
                fwd = self.model(
                    input_ids=delta_ids["input_ids"],
                    past_key_values=self._past,
                    use_cache=True,
                )
                next_id     = int(torch.argmax(fwd.logits[:, -1, :], dim=-1)[0].item())
                result_text = self.tok.decode([next_id])
                self._past  = fwd.past_key_values

        step_s = time.perf_counter() - t0

        self._step      += 1
        self._sp_tokens += delta_tokens
        self._total_s   += step_s

        # Naive token count (base + all deltas so far)
        self._naive_tokens += self._base_tokens + delta_tokens

        # Emit receipt
        self.ps.infer(self._current_sha256, delta_text, delta_tokens, self._base_tokens)

        # Optional merge
        if self.merge_every > 0 and self._step % self.merge_every == 0:
            new_tokens = self._base_tokens + delta_tokens
            self.ps.merge(self._current_sha256, delta_text,
                          from_dynamic_cache(self._past), new_tokens, dtype=self.dtype)
            merged_pre           = f"{self._current_sha256}|{_sha256(delta_text)}"
            self._current_sha256 = _sha256(merged_pre)
            self._base_tokens    = new_tokens

        return result_text

    def stats(self) -> dict:
        saved = self._naive_tokens - self._sp_tokens
        return {
            "steps":        self._step,
            "naive_tokens": self._naive_tokens,
            "sp_tokens":    self._sp_tokens,
            "tokens_saved": saved,
            "savings_pct":  round(saved / self._naive_tokens * 100, 2) if self._naive_tokens else 0,
            "total_s":      round(self._total_s, 3),
        }

    # ------------------------------------------------------------------
    # LangChain compatibility shim
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "state-pack"

    def _call(self, prompt: str, stop=None) -> str:
        """LangChain BaseLLM interface."""
        return self(prompt)

    def _generate(self, prompts: List[str], stop=None):
        """LangChain BaseLLM generate interface."""
        try:
            from langchain.schema import LLMResult, Generation
            gens = [[Generation(text=self(p))] for p in prompts]
            return LLMResult(generations=gens)
        except ImportError:
            return [self(p) for p in prompts]
