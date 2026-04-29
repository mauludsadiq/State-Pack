#!/usr/bin/env python3
import argparse
import json
from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--text-file")
parser.add_argument("--text")
args = parser.parse_args()

if args.text is not None:
    text = args.text
elif args.text_file is not None:
    text = open(args.text_file, "r", encoding="utf-8").read()
else:
    raise SystemExit("required: --text or --text-file")
tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok(text, add_special_tokens=False)["input_ids"]

print(json.dumps({
    "token_count": len(ids),
    "token_ids": ids
}))
