#!/usr/bin/env python3
"""
state_pack.server
~~~~~~~~~~~~~~~~~
FastAPI HTTP server for State Pack.
Exposes create / infer / merge / verify / resolve over REST.

Start: PYTHONPATH=. python3 -m state_pack.server --store demo/api_store --model gpt2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .client import StatePack, _sha256
from .serialize import load_kv_cache, save_kv_cache

app = FastAPI(title="State Pack API", version="0.1.0")

# Global state — set at startup
_sp: Optional[StatePack] = None
_model = None
_tok   = None
_model_id: str = "gpt2"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateRequest(BaseModel):
    base_text: str
    model_id:  Optional[str] = None   # override global model

class InferRequest(BaseModel):
    base_sha256: str
    delta_text:  str

class MergeRequest(BaseModel):
    base_sha256: str
    delta_text:  str

class VerifyRequest(BaseModel):
    base_sha256: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "model": _model_id, "store": str(_sp.store)}


@app.post("/packets")
def create(req: CreateRequest):
    """
    Run base_text through the loaded model, serialize KV cache, register packet.
    Returns receipt + base_sha256.
    """
    tok, model = _tok, _model
    ids = tok(req.base_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(**ids, use_cache=True)

    base_tokens = ids["input_ids"].shape[1]
    receipt = _sp.create(req.base_text, out.past_key_values, base_tokens)
    receipt["base_sha256"] = _sha256(req.base_text)
    receipt["base_tokens"] = base_tokens
    return receipt


@app.post("/infer")
def infer(req: InferRequest):
    """
    Load KV cache for base_sha256, run delta_text through model,
    emit infer receipt. Returns receipt + next_token sample.
    """
    try:
        blob = _sp.load_base(req.base_sha256)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    past = blob["past_key_values"]
    tok, model = _tok, _model

    delta_ids = tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )

    next_token_id  = int(torch.argmax(out.logits[:, -1, :], dim=-1)[0].item())
    next_token_str = tok.decode([next_token_id])

    receipt = _sp.infer(req.base_sha256, req.delta_text)
    receipt["next_token"] = {"id": next_token_id, "text": next_token_str}
    return receipt


@app.post("/merge")
def merge(req: MergeRequest):
    """
    Load KV cache for base_sha256, run delta through model,
    save merged KV cache as new packet. Returns receipt.
    """
    try:
        blob = _sp.load_base(req.base_sha256)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    past = blob["past_key_values"]
    tok, model = _tok, _model

    delta_ids = tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )

    new_tokens = blob["base_tokens"] + delta_ids["input_ids"].shape[1]
    receipt = _sp.merge(
        req.base_sha256,
        req.delta_text,
        out.past_key_values,
        new_tokens,
    )
    return receipt


@app.post("/verify")
def verify(req: VerifyRequest):
    try:
        return _sp.verify(req.base_sha256)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/packets/{base_sha256}")
def resolve(base_sha256: str):
    try:
        info = _sp.resolve(base_sha256)
        manifest = json.loads(Path(info["manifest"]).read_text())
        return {"ok": True, **manifest}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _load_model(model_id: str):
    global _model, _tok, _model_id
    print(f"Loading model {model_id}...")
    _tok     = AutoTokenizer.from_pretrained(model_id)
    _model   = AutoModelForCausalLM.from_pretrained(model_id)
    _model.eval()
    _model_id = model_id
    print("Model ready.")


def main():
    global _sp
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="packets")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=8000)
    args = parser.parse_args()

    _load_model(args.model)
    _sp = StatePack(store=args.store, model_id=args.model)
    print(f"Store: {_sp.store}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
