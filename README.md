# State Pack

A **content-addressed transformer state protocol** for efficient agent loops.

## Results

| Benchmark | Token Savings | Speedup | Latency |
|-----------|--------------|---------|---------|
| SDK agent loop (40 steps, GPT-2) | 95.3% | 3.96x | - |
| HTTP API (40 steps, GPT-2) | 60.9% | - | 43ms |

## How It Works

1. **CREATE** - Run base prompt through model, serialize KV cache to content-addressed blob
2. **INFER** - Load cached KV state, process delta tokens only, emit verifiable receipt
3. **MERGE** - Fold accumulated deltas back into base when threshold is reached

Every artifact is SHA-256 addressed. Every operation emits a tamper-evident receipt.

## Quickstart

### Python SDK

```python
from state_pack import StatePack
from state_pack.client import _sha256
import torch

sp = StatePack(store='my_store', model_id='gpt2')

# Base pass
receipt = sp.create(base_text, out.past_key_values, base_tokens)

# Delta steps - only new tokens processed
receipt = sp.infer(_sha256(base_text), delta_text)
print(receipt['tokens']['saved'], 'tokens saved')
```

### Agent Loop

```python
from state_pack.agent_loop import AgentLoop

loop = AgentLoop(model, tok, store='my_store', model_id='gpt2', merge_every=10)
results = loop.run(base_text, deltas)
# tokens_saved: 17785, savings_pct: 95.31, speedup: 3.958
```

### HTTP API

```bash
PYTHONPATH=. python3 -m state_pack.server --store my_store --model gpt2

curl -X POST http://localhost:8000/packets \
  -H 'Content-Type: application/json' \
  -d '{"base_text": "You are a research agent..."}'

curl -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{"base_sha256": "<sha>", "delta_text": "Step 1: observe clause A."}'
```

### CLI

```bash
cargo run -- create --model gpt2 --base base.txt --blob blob.pt --out store/
cargo run -- verify --manifest store/state_packet_<hash>.json
cargo run -- benchmark-native --base base.txt --blob blob.pt --steps 40
```

## Architecture

```
state_pack/
  serialize.py   KV cache to .pt blob serialization
  store.py       In-process packet store (no subprocess, 43ms/step)
  client.py      High-level SDK (uses Rust CLI for CLI workflows)
  agent_loop.py  Drop-in agent loop with automatic KV reuse
  server.py      FastAPI HTTP server

src/main.rs      Rust CLI - content addressing, receipts, benchmarks
```

## Model Support

| Model | Status |
|-------|--------|
| GPT-2 | Verified |
| Llama (tiny) | Verified |
| Any HuggingFace CausalLM | Should work |

## Roadmap

- [x] Phase 1 - Python SDK (serialize, client, agent_loop)
- [x] Phase 2 - HTTP API (FastAPI, PacketStore, 43ms/step)
- [ ] Phase 3 - KV cache portability across devices (float16, quantization)
- [ ] Phase 4 - Framework integration (LangChain, LangGraph)

## License

MUI
