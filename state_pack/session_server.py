from __future__ import annotations

"""
state_pack.session_server
~~~~~~~~~~~~~~~~~~~~~~~~~
Session-aware inference server. KV cache stays in memory per session.
Identical base prompts share a single cached state across all agents.

POST /sessions          - register base prompt, get session_id
POST /sessions/{id}/step - run delta against in-memory KV cache
DELETE /sessions/{id}   - release session
GET  /sessions          - list active sessions + stats
GET  /health            - server stats
"""

import hashlib
import time
import uuid
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .serialize import to_dynamic_cache, from_dynamic_cache, save_kv_cache
from .store import PacketStore, _sha256

app = FastAPI(title="State Pack Session Server", version="0.1.0")

_model     = None
_tok       = None
_model_id  = "gpt2"
_ps: Optional[PacketStore] = None

# In-memory session store
# base_sha256 -> DynamicCache (shared across all sessions with same base)
_base_cache: Dict[str, DynamicCache] = {}
_base_tokens: Dict[str, int] = {}

# session_id -> session state
_sessions: Dict[str, dict] = {}


class CreateSessionRequest(BaseModel):
    base_text: str
    ttl_seconds: int = 3600

class StepRequest(BaseModel):
    delta_text: str
    generate: bool = False
    max_new_tokens: int = 32


def _get_or_create_base(base_text: str) -> tuple:
    base_sha256 = _sha256(base_text)

    if base_sha256 not in _base_cache:
        ids = _tok(base_text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = _model(**ids, use_cache=True)
        base_tokens = ids["input_ids"].shape[1]
        _base_cache[base_sha256]  = to_dynamic_cache(out.past_key_values)
        _base_tokens[base_sha256] = base_tokens

        # Persist to store
        _ps.create(base_text, out.past_key_values, base_tokens, dtype=torch.float16)
        created = True
    else:
        created = False

    return base_sha256, _base_cache[base_sha256], _base_tokens[base_sha256], created


@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    t0 = time.perf_counter()
    base_sha256, base_kv, base_tokens, created = _get_or_create_base(req.base_text)
    elapsed = time.perf_counter() - t0

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id":   session_id,
        "base_sha256":  base_sha256,
        "base_tokens":  base_tokens,
        "past":         base_kv,      # shared reference - do NOT mutate
        "step":         0,
        "naive_tokens": 0,
        "sp_tokens":    base_tokens,
        "created_at":   time.time(),
        "expires_at":   time.time() + req.ttl_seconds,
        "cache_hit":    not created,
    }

    return {
        "session_id":  session_id,
        "base_sha256": base_sha256,
        "base_tokens":  base_tokens,
        "cache_hit":   not created,
        "elapsed_s":   round(elapsed, 3),
        "active_sessions": len(_sessions),
        "shared_bases":    len(_base_cache),
    }


@app.post("/sessions/{session_id}/step")
def step(session_id: str, req: StepRequest):
    if session_id not in _sessions:
        raise HTTPException(404, detail=f"Session {session_id} not found")

    sess = _sessions[session_id]

    if time.time() > sess["expires_at"]:
        del _sessions[session_id]
        raise HTTPException(410, detail="Session expired")

    delta_ids    = _tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    delta_tokens = delta_ids["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        fwd = _model(
            input_ids=delta_ids["input_ids"],
            past_key_values=sess["past"],
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    next_id  = int(torch.argmax(fwd.logits[:, -1, :], dim=-1)[0].item())
    next_tok = _tok.decode([next_id])

    # Update session state with new past (private copy after first step)
    sess["past"]  = fwd.past_key_values
    sess["step"] += 1

    naive_this_step    = sess["base_tokens"] + delta_tokens * sess["step"]
    sess["naive_tokens"] += naive_this_step
    sess["sp_tokens"]    += delta_tokens

    saved   = sess["naive_tokens"] - sess["sp_tokens"]
    savings = round(saved / sess["naive_tokens"] * 100, 2) if sess["naive_tokens"] else 0

    # Receipt
    receipt = _ps.infer(sess["base_sha256"], req.delta_text, delta_tokens, sess["base_tokens"])

    return {
        "session_id":    session_id,
        "step":          sess["step"],
        "delta_tokens":  delta_tokens,
        "base_tokens":   sess["base_tokens"],
        "next_token":    {"id": next_id, "text": next_tok},
        "elapsed_ms":    round(elapsed * 1000, 1),
        "receipt_id":    receipt["receipt_id"],
        "savings": {
            "naive_tokens":  sess["naive_tokens"],
            "sp_tokens":     sess["sp_tokens"],
            "tokens_saved":  saved,
            "savings_pct":   savings,
        },
    }


@app.delete("/sessions/{session_id}")
def close_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, detail="Session not found")
    sess = _sessions.pop(session_id)
    saved = sess["naive_tokens"] - sess["sp_tokens"]
    return {
        "session_id":   session_id,
        "steps":        sess["step"],
        "tokens_saved": saved,
        "savings_pct":  round(saved / sess["naive_tokens"] * 100, 2) if sess["naive_tokens"] else 0,
    }


@app.get("/sessions")
def list_sessions():
    now = time.time()
    active = {
        sid: {
            "step":        s["step"],
            "base_sha256": s["base_sha256"][:16] + "...",
            "base_tokens": s["base_tokens"],
            "sp_tokens":   s["sp_tokens"],
            "expires_in":  round(s["expires_at"] - now, 0),
            "cache_hit":   s["cache_hit"],
        }
        for sid, s in _sessions.items()
        if now <= s["expires_at"]
    }
    return {
        "active_sessions": len(active),
        "shared_bases":    len(_base_cache),
        "sessions":        active,
    }


@app.get("/health")
def health():
    return {
        "ok":             True,
        "model":          _model_id,
        "active_sessions": len(_sessions),
        "shared_bases":   len(_base_cache),
        "store":          str(_ps.store),
    }


def main():
    import argparse
    global _model, _tok, _model_id, _ps
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="packets")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=8001)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    _tok    = AutoTokenizer.from_pretrained(args.model)
    _model  = AutoModelForCausalLM.from_pretrained(args.model)
    _model.eval()
    _model_id = args.model
    _ps = PacketStore(store=args.store, model_id=args.model)
    print(f"Ready. Store: {_ps.store}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
