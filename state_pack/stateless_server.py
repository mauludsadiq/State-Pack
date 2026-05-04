from __future__ import annotations

"""
state_pack.stateless_server
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure stateless inference server.
Server is a content-addressed KV blob store + inference engine.
Zero session state. Client owns the hash chain.

POST /states           - Create base state from text
POST /infer            - Run inference from a state hash
POST /merge            - Create new state from base + delta (no inference)
POST /compact          - Fold accumulated deltas into fresh base state
GET  /states/{hash}    - Inspect a cached state
GET  /health           - Server status
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .store import PacketStore, _sha256
from .serialize import save_kv_cache, load_kv_cache, to_dynamic_cache, from_dynamic_cache

app = FastAPI(title="State Pack Stateless API", version="0.2.0")

_model     = None
_tok       = None
_model_id  = "gpt2"
_ps: Optional[PacketStore] = None

# In-memory KV index: state_hash -> DynamicCache
# Server caches hot states in memory; cold states loaded from disk
_hot: dict = {}
_HOT_MAX = 64  # max in-memory states


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateStateRequest(BaseModel):
    base_text: str
    dtype: str = "float16"

class InferRequest(BaseModel):
    state_hash: str
    delta_text: str
    dtype: str = "float16"

class MergeRequest(BaseModel):
    state_hash: str
    delta_text: str
    dtype: str = "float16"

class CompactRequest(BaseModel):
    state_hash: str
    accumulated_deltas: list[str]
    dtype: str = "float16"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float16)


def _get_kv(state_hash: str) -> DynamicCache:
    if state_hash in _hot:
        return _hot[state_hash]
    try:
        blob = _ps.load_kv(state_hash)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"State not found: {state_hash}")
    raw = tuple(tuple(t.to(torch.float32) for t in l) for l in blob["past_key_values"])
    cache = to_dynamic_cache(raw)
    _hot[state_hash] = cache
    return cache


def _put_kv(state_hash: str, past_key_values, base_text: str,
             base_tokens: int, dtype: torch.dtype) -> dict:
    if len(_hot) >= _HOT_MAX:
        oldest = next(iter(_hot))
        del _hot[oldest]

    _ps.create(base_text, past_key_values, base_tokens, dtype=dtype)

    raw = tuple(tuple(t.to(torch.float32) for t in l) for l in past_key_values)
    _hot[state_hash] = to_dynamic_cache(raw)

    blob_path = _ps.store / f"state_packet_{state_hash}.pt"
    return {
        "state_hash": state_hash,
        "tokens": base_tokens,
        "bytes": blob_path.stat().st_size if blob_path.exists() else 0,
    }


def _receipt(op: str, **kwargs) -> dict:
    r = {"op": op, "ok": True, **kwargs}
    canonical = json.dumps(r, sort_keys=True, separators=(",", ":")).encode()
    r["receipt_id"] = "sha256:" + _sha256(canonical)
    return r


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": _model_id,
        "store": str(_ps.store),
        "states_hot": len(_hot),
        "states_cached": len(list(_ps.store.glob("*.json"))),
    }


@app.post("/states")
def create_state(req: CreateStateRequest):
    """
    Run base_text through model, serialize KV cache, return state_hash.
    Idempotent: same text always returns same hash.
    """
    t0 = time.perf_counter()
    state_hash = _sha256(req.base_text)
    cache_hit  = state_hash in _hot or (_ps.store / f"state_packet_{state_hash}.pt").exists()

    if not cache_hit:
        ids = _tok(req.base_text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = _model(**ids, use_cache=True)
        base_tokens = ids["input_ids"].shape[1]
        info = _put_kv(state_hash, out.past_key_values, req.base_text,
                       base_tokens, _dtype(req.dtype))
    else:
        _get_kv(state_hash)  # warm the hot cache
        manifest = _ps._load_manifest(state_hash)
        info = {
            "state_hash": state_hash,
            "tokens": manifest["base_tokens"],
            "bytes": manifest["blob_bytes"],
        }

    elapsed = time.perf_counter() - t0
    return _receipt(
        "create",
        state_hash=state_hash,
        tokens=info["tokens"],
        bytes=info["bytes"],
        cache_hit=cache_hit,
        elapsed_s=round(elapsed, 3),
    )


@app.post("/infer")
def infer(req: InferRequest):
    """
    (state_hash, delta_text) -> (new_state_hash, output)
    Pure function. Server holds no session state.
    Client chains: h0 -> infer -> h1 -> infer -> h2
    """
    t0   = time.perf_counter()
    past = _get_kv(req.state_hash)

    manifest     = _ps._load_manifest(req.state_hash)
    base_tokens  = manifest["base_tokens"]
    base_bytes   = manifest["base_bytes"]

    delta_ids    = _tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    delta_tokens = delta_ids["input_ids"].shape[1]

    with torch.no_grad():
        out = _model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )

    next_id  = int(torch.argmax(out.logits[:, -1, :], dim=-1)[0].item())
    next_tok = _tok.decode([next_id])

    # New state hash = hash of (base_hash | delta_hash)
    delta_hash    = _sha256(req.delta_text)
    new_state_hash = _sha256(f"{req.state_hash}|{delta_hash}")
    new_tokens     = base_tokens + delta_tokens

    # Store new state
    new_base_text = f"{req.state_hash}|{delta_hash}"
    _put_kv(new_state_hash, from_dynamic_cache(out.past_key_values),
             new_base_text, new_tokens, _dtype(req.dtype))

    elapsed = time.perf_counter() - t0

    return _receipt(
        "infer",
        state_hash=req.state_hash,
        new_state_hash=new_state_hash,
        delta_hash=delta_hash,
        output=next_tok,
        tokens={
            "base": base_tokens,
            "delta": delta_tokens,
            "new_total": new_tokens,
            "saved": base_tokens,
            "savings_pct": round(base_tokens / (base_tokens + delta_tokens) * 100, 2),
        },
        elapsed_ms=round(elapsed * 1000, 1),
    )


@app.post("/merge")
def merge(req: MergeRequest):
    """
    Create new state from base + delta without running inference.
    Use when you want to advance the state chain without generating output.
    """
    t0       = time.perf_counter()
    past     = _get_kv(req.state_hash)
    manifest = _ps._load_manifest(req.state_hash)

    delta_ids    = _tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    delta_tokens = delta_ids["input_ids"].shape[1]

    with torch.no_grad():
        out = _model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )

    delta_hash     = _sha256(req.delta_text)
    new_state_hash = _sha256(f"{req.state_hash}|{delta_hash}")
    new_tokens     = manifest["base_tokens"] + delta_tokens
    new_base_text  = f"{req.state_hash}|{delta_hash}"

    _put_kv(new_state_hash, from_dynamic_cache(out.past_key_values),
             new_base_text, new_tokens, _dtype(req.dtype))

    elapsed = time.perf_counter() - t0
    return _receipt(
        "merge",
        state_hash=req.state_hash,
        new_state_hash=new_state_hash,
        delta_hash=delta_hash,
        tokens=new_tokens,
        elapsed_ms=round(elapsed * 1000, 1),
    )


@app.post("/compact")
def compact(req: CompactRequest):
    """
    Fold accumulated deltas into a fresh base state.
    Client provides the list of deltas it has accumulated.
    Server recomputes the full state from base + all deltas once,
    returns a new canonical state hash. Client discards intermediate hashes.
    """
    t0       = time.perf_counter()
    past     = _get_kv(req.state_hash)
    manifest = _ps._load_manifest(req.state_hash)

    total_delta_tokens = 0
    current_past = past

    for delta_text in req.accumulated_deltas:
        delta_ids = _tok(delta_text, return_tensors="pt", add_special_tokens=False)
        total_delta_tokens += delta_ids["input_ids"].shape[1]
        with torch.no_grad():
            out = _model(
                input_ids=delta_ids["input_ids"],
                past_key_values=current_past,
                use_cache=True,
            )
        current_past = out.past_key_values

    # New compacted state: hash of (base_hash | all_deltas_joined)
    joined         = "|".join(req.accumulated_deltas)
    compact_hash   = _sha256(f"{req.state_hash}|compact|{_sha256(joined)}")
    new_tokens     = manifest["base_tokens"] + total_delta_tokens
    compact_text   = f"compact:{req.state_hash}+{len(req.accumulated_deltas)}deltas"

    _put_kv(compact_hash, from_dynamic_cache(current_past),
             compact_text, new_tokens, _dtype(req.dtype))

    elapsed = time.perf_counter() - t0
    return _receipt(
        "compact",
        state_hash=req.state_hash,
        new_state_hash=compact_hash,
        steps_folded=len(req.accumulated_deltas),
        tokens=new_tokens,
        delta_tokens_folded=total_delta_tokens,
        elapsed_ms=round(elapsed * 1000, 1),
    )


@app.get("/states/{state_hash}")
def get_state(state_hash: str):
    """Inspect a cached state."""
    try:
        manifest = _ps._load_manifest(state_hash)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"State not found: {state_hash}")
    return {
        "ok": True,
        "state_hash": state_hash,
        "tokens": manifest["base_tokens"],
        "bytes": manifest["blob_bytes"],
        "model": manifest["model"],
        "hot": state_hash in _hot,
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    global _model, _tok, _model_id, _ps
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="packets")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=8002)
    parser.add_argument("--hot-max", type=int, default=64)
    args = parser.parse_args()

    global _HOT_MAX
    _HOT_MAX = args.hot_max

    print(f"Loading {args.model}...")
    _tok    = AutoTokenizer.from_pretrained(args.model)
    _model  = AutoModelForCausalLM.from_pretrained(args.model)
    _model.eval()
    _model_id = args.model
    _ps = PacketStore(store=args.store, model_id=args.model)
    print(f"Ready. Store: {_ps.store} | {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
