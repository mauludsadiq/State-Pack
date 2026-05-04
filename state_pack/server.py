from __future__ import annotations

import argparse
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .store import PacketStore
from .store import _sha256

app = FastAPI(title="State Pack API", version="0.1.0")

_ps: Optional[PacketStore] = None
_model = None
_tok   = None
_model_id: str = "gpt2"


class CreateRequest(BaseModel):
    base_text: str

class InferRequest(BaseModel):
    base_sha256: str
    delta_text:  str

class MergeRequest(BaseModel):
    base_sha256: str
    delta_text:  str

class VerifyRequest(BaseModel):
    base_sha256: str


@app.get("/health")
def health():
    return {"ok": True, "model": _model_id, "store": str(_ps.store)}


@app.post("/packets")
def create(req: CreateRequest):
    ids = _tok(req.base_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = _model(**ids, use_cache=True)
    base_tokens = ids["input_ids"].shape[1]
    receipt = _ps.create(req.base_text, out.past_key_values, base_tokens, dtype=torch.float16)
    receipt["base_sha256"] = _sha256(req.base_text)
    receipt["base_tokens"] = base_tokens
    return receipt


@app.post("/infer")
def infer(req: InferRequest):
    try:
        blob = _ps.load_kv(req.base_sha256)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))

    from state_pack.serialize import to_dynamic_cache
    _pkv = blob["past_key_values"]
    past = to_dynamic_cache(tuple(tuple(t.to(torch.float32) for t in l) for l in _pkv))
    base_tokens = blob["base_tokens"]

    delta_ids = _tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = _model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )

    delta_tokens   = delta_ids["input_ids"].shape[1]
    next_token_id  = int(torch.argmax(out.logits[:, -1, :], dim=-1)[0].item())
    next_token_str = _tok.decode([next_token_id])

    receipt = _ps.infer(req.base_sha256, req.delta_text, delta_tokens, base_tokens)
    receipt["next_token"] = {"id": next_token_id, "text": next_token_str}
    return receipt


@app.post("/merge")
def merge(req: MergeRequest):
    try:
        blob = _ps.load_kv(req.base_sha256)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))

    past        = blob["past_key_values"]
    base_tokens = blob["base_tokens"]

    delta_ids = _tok(req.delta_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = _model(
            input_ids=delta_ids["input_ids"],
            past_key_values=past,
            use_cache=True,
        )

    new_tokens = base_tokens + delta_ids["input_ids"].shape[1]
    return _ps.merge(req.base_sha256, req.delta_text, out.past_key_values, new_tokens, dtype=torch.float16)


@app.post("/verify")
def verify(req: VerifyRequest):
    try:
        return _ps.verify(req.base_sha256)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))


@app.get("/packets/{base_sha256}")
def resolve(base_sha256: str):
    try:
        m = _ps._load_manifest(base_sha256)
        return {"ok": True, **m}
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))


def main():
    global _ps, _model, _tok, _model_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="packets")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    _tok     = AutoTokenizer.from_pretrained(args.model)
    _model   = AutoModelForCausalLM.from_pretrained(args.model)
    _model.eval()
    _model_id = args.model
    _ps = PacketStore(store=args.store, model_id=args.model)
    print(f"Store: {_ps.store}  |  Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()