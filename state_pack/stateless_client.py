from __future__ import annotations
import hashlib, json, pathlib, time, urllib.request
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from .compaction import CompactionPolicy, DEFAULT_POLICY
from .serialize import save_kv_cache, to_dynamic_cache, from_dynamic_cache
from .store import _sha256

class StatelessClient:
    def __init__(self, model_id="gpt2", rust_server="http://localhost:8003",
                 store="packets", policy=None, dtype=torch.float16):
        self.model_id  = model_id
        self.server    = rust_server.rstrip("/")
        self.store     = pathlib.Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.policy    = policy or DEFAULT_POLICY
        self.dtype     = dtype
        print(f"Loading {model_id}...")
        self.tok   = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self._state_hash   = None
        self._base_tokens  = 0
        self._past         = None
        self._step         = 0
        self._deltas       = []
        self._naive_tokens = 0
        self._sp_tokens    = 0
        self._compact_count = 0
        self._total_s      = 0.0

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(f"{self.server}{path}", data=data,
               headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def _get(self, path):
        with urllib.request.urlopen(f"{self.server}{path}") as r:
            return json.loads(r.read())

    def set_base(self, base_text):
        base_sha256 = _sha256(base_text)
        blob_path   = self.store / f"state_packet_{base_sha256}.pt"
        ids = self.tok(base_text, return_tensors="pt", add_special_tokens=False)
        t0  = time.perf_counter()
        with torch.no_grad():
            out = self.model(**ids, use_cache=True)
        base_s = time.perf_counter() - t0
        base_tokens = ids["input_ids"].shape[1]
        save_kv_cache(blob_path, out.past_key_values, self.model_id,
                      base_text, base_tokens, dtype=self.dtype)
        self._past        = to_dynamic_cache(out.past_key_values)
        self._state_hash  = base_sha256
        self._base_tokens = base_tokens
        self._step        = 0
        self._deltas      = []
        self._sp_tokens   = base_tokens
        self._total_s     = base_s
        return self._post("/states", {"state_hash": base_sha256,
                                      "base_tokens": base_tokens,
                                      "base_text": base_text})

    def step(self, delta_text):
        if self._past is None:
            raise RuntimeError("Call set_base() first.")
        delta_ids    = self.tok(delta_text, return_tensors="pt", add_special_tokens=False)
        delta_tokens = delta_ids["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = self.model(input_ids=delta_ids["input_ids"],
                             past_key_values=self._past, use_cache=True)
        step_s = time.perf_counter() - t0
        self._past = fwd.past_key_values
        next_id = int(torch.argmax(fwd.logits[:, -1, :], dim=-1)[0].item())
        output  = self.tok.decode([next_id])
        delta_sha256   = _sha256(delta_text)
        new_state_hash = _sha256(f"{self._state_hash}|{delta_sha256}")
        new_tokens     = self._base_tokens + delta_tokens
        new_blob = self.store / f"state_packet_{new_state_hash}.pt"
        save_kv_cache(new_blob, from_dynamic_cache(self._past), self.model_id,
                      f"{self._state_hash}|{delta_sha256}", new_tokens, dtype=self.dtype)
        receipt = self._post("/infer", {"state_hash": self._state_hash,
                                        "delta_text": delta_text,
                                        "delta_tokens": delta_tokens})
        self._post("/states", {"state_hash": new_state_hash, "base_tokens": new_tokens})
        self._state_hash   = new_state_hash
        self._base_tokens  = new_tokens
        self._step        += 1
        self._deltas.append(delta_text)
        self._sp_tokens   += delta_tokens
        self._naive_tokens += new_tokens
        self._total_s     += step_s
        savings_pct = receipt["tokens"]["savings_pct"]
        compacted = False
        if self.policy.should_compact(self._base_tokens, self._deltas, self._step):
            self._compact()
            compacted = True
        return {"step": self._step, "output": output, "state_hash": new_state_hash,
                "savings_pct": savings_pct, "elapsed_ms": round(step_s * 1000, 1),
                "compacted": compacted, "receipt_id": receipt["receipt_id"]}

    def _compact(self):
        if not self._deltas: return
        joined       = "|".join(self._deltas)
        compact_hash = _sha256(f"{self._state_hash}|compact|{_sha256(joined)}")
        new_tokens   = self._base_tokens
        compact_blob = self.store / f"state_packet_{compact_hash}.pt"
        save_kv_cache(compact_blob, from_dynamic_cache(self._past), self.model_id,
                      f"compact:{self._state_hash}", new_tokens, dtype=self.dtype)
        self._post("/compact", {"state_hash": self._state_hash,
                                "new_state_hash": compact_hash,
                                "new_blob_path": str(compact_blob),
                                "new_base_tokens": new_tokens,
                                "steps_folded": len(self._deltas)})
        self._state_hash    = compact_hash
        self._deltas        = []
        self._compact_count += 1
        self.policy.reset()

    def stats(self):
        saved = self._naive_tokens - self._sp_tokens
        return {"steps": self._step, "naive_tokens": self._naive_tokens,
                "sp_tokens": self._sp_tokens, "tokens_saved": saved,
                "savings_pct": round(saved / self._naive_tokens * 100, 2) if self._naive_tokens else 0,
                "compact_count": self._compact_count, "total_s": round(self._total_s, 3),
                "avg_ms_per_step": round(self._total_s / self._step * 1000, 1) if self._step else 0}

    def health(self):
        return self._get("/health")
