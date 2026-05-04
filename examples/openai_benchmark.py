#!/usr/bin/env python3
import json
from state_pack.openai_integration import run_benchmark

results = run_benchmark(steps=20)

print()
print("=" * 50)
print(f"  Model:   {results['model']}")
print(f"  Steps:   {results['steps']}")
print("=" * 50)
print(f"  Naive   input tokens: {results['naive']['input_tokens']:,}")
print(f"  StatePk input tokens: {results['state_pack']['input_tokens']:,}")
print(f"  Tokens saved:         {results['savings']['input_tokens_saved']:,} ({results['savings']['savings_pct']}%)")
print(f"  Naive cost:           ${results['naive']['cost_usd']:.6f}")
print(f"  StatePk cost:         ${results['state_pack']['cost_usd']:.6f}")
print(f"  Cost saved:           ${results['savings']['cost_saved_usd']:.6f} ({results['savings']['cost_reduction_pct']}%)")
print("=" * 50)
print()
print(json.dumps(results, indent=2))
