#!/usr/bin/env python3
import argparse
import json
import hashlib
from transformers import GPT2Tokenizer

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True)
args = parser.parse_args()

tok = GPT2Tokenizer.from_pretrained("gpt2")
token_ids = tok(args.text, add_special_tokens=False)["input_ids"]

payload = {
    "model": "gpt2",
    "delta_sha256": sha256_text(args.text),
    "token_count": len(token_ids),
    "token_ids": token_ids,
}

payload["token_trace_sha256"] = hashlib.sha256(
    json.dumps(token_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()

print(json.dumps(payload, indent=2))
