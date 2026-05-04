# State Pack

**The CDN for AI inference costs.**

Agents pay per token. State Pack makes that cost invisible — the same way
BlackBerry made per-character SMS costs invisible. Not by changing the model.
Not by changing the API. By caching state at the infrastructure layer.

## Benchmarks

| Model | Params | Token Savings | Speedup | Blob Size |
|-------|--------|--------------|---------|-----------|
| GPT-2 | 124M | 95.3% | 3.96x | 0.5MB |
| Qwen2.5-3B | 3B | 90.85% | 1.35x* | 1.2MB |
| Mistral-7B-Instruct | 7B | 90.92% | 1.42x* | 5.7MB |
| OpenAI API (gpt-4o-mini) | — | 92.64% | — | — |

*CPU only. GPU expected 3-4x based on GPT-2 results.

Token savings are architecture-independent and model-size-independent.
90-95% across 124M to 7B parameters. The savings are structural — they
come from the protocol, not the model.

## Proven on the OpenAI API

| Metric | Naive | State Pack | Saving |
|--------|-------|------------|--------|
| Input tokens (20-step loop) | 17,929 | 1,320 | 92.6% |
| Cost per loop (gpt-4o-mini) | $0.00341 | $0.00091 | 73.4% |
| Cost per loop (gpt-4o) | ~$0.180 | ~$0.048 | 73.4% |
| Latency per step | — | 50ms | — |
| Base cache hit (shared agents) | 0.951s | 0.003s | 99% |

Real numbers. Real API. Reproducible with one command on your own key.

## The Math at Scale

1,000 agents. 40-step loops. GPT-4o pricing.

| | Naive | State Pack |
|--|-------|------------|
| Cost per cycle | $144.40 | $36.32 |
| Saving per cycle | | $108.08 |
| At 100 cycles/day | $14,440 | $3,632 |
| Daily saving | | $10,808 |

If 1,000 agents share the same system prompt,
the base KV cache is computed once and served to all.
Agents 2-1000 pay 0 tokens for context setup.

## How It Works

```
naive:       [system + history + delta] -> model   (cost grows every step)
state pack:  [delta only]               -> model   (cost stays flat)
```

1. CREATE  - run base prompt once, serialize KV cache to content-addressed blob
2. INFER   - load cached state, process delta tokens only, emit verifiable receipt
3. COMPACT - fold accumulated deltas into fresh base (keeps savings compounding)

Every artifact is SHA-256 addressed. Every operation emits a tamper-evident receipt.
Same inputs always produce same outputs. Fully auditable.

## v0.2: Stateless Inference Protocol

The server is a pure function. Zero session state. Client owns the hash chain.

```
POST /states           -> { state_hash }
POST /infer            -> { new_state_hash, output, savings }
POST /merge            -> { new_state_hash }
POST /compact          -> { new_state_hash, steps_folded }
GET  /states/{hash}    -> { tokens, bytes, hot }
GET  /health           -> { states_hot, states_cached }
```

Client chains hashes:

```
h0 -> infer -> h1 -> infer -> h2 -> ... -> compact -> h_fresh
```

Server never knows what a conversation is.
Same state_hash from any client always returns the same result.
Infinitely horizontally scalable.

## Quickstart

### Prove it against your own OpenAI key

```bash
git clone https://github.com/mauludsadiq/State-Pack.git
cd State-Pack
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python3 examples/openai_benchmark.py
```

### Stateless server (v0.2)

```bash
pip install state-pack
PYTHONPATH=. python3 -m state_pack.stateless_server --store my_store --model gpt2

# Create base state (idempotent)
curl -X POST http://localhost:8002/states \
  -H 'Content-Type: application/json' \
  -d '{"base_text": "You are a legal research agent..."}'
# -> {"state_hash": "sha256:abc..."}

# Infer - pure function
curl -X POST http://localhost:8002/infer \
  -H 'Content-Type: application/json' \
  -d '{"state_hash": "sha256:abc...", "delta_text": "Step 1: clause affects indemnity."}'
# -> {"new_state_hash": "sha256:def...", "output": "...", "savings": {...}}

# Compact accumulated deltas into fresh base
curl -X POST http://localhost:8002/compact \
  -H 'Content-Type: application/json' \
  -d '{"state_hash": "sha256:abc...", "accumulated_deltas": ["Step 1...", "Step 2..."]}'
# -> {"new_state_hash": "sha256:ghi...", "steps_folded": 2}
```

### Python SDK

```python
from state_pack.llm import StatePackLLM

llm = StatePackLLM.from_pretrained('gpt2', store='my_store', merge_every=10)
llm.set_base('You are a research agent...\n\n')

for delta in steps:
    output = llm(delta)

print(llm.stats())
# tokens_saved: 17785, savings_pct: 95.31, speedup: 3.958
```

## Architecture

```
state_pack/
  stateless_server.py  Pure stateless protocol (v0.2) - hash chain API
  session_server.py    In-memory KV cache, base dedup, 1000-agent scale
  server.py            HTTP API (FastAPI, 43ms/step)
  llm.py               Drop-in LLM wrapper with automatic KV reuse
  store.py             In-process packet store (no subprocess)
  serialize.py         KV cache to .pt blob (float16, 50% smaller)
  client.py            High-level SDK
  agent_loop.py        Drop-in agent loop
  openai_integration.py  Benchmark against OpenAI API

src/main.rs            Rust CLI - content addressing, receipts, protocol
```

## Model Support

| Model | Status |
|-------|--------|
| GPT-2 (124M) | Verified |
| Qwen2.5-3B | Verified |
| Mistral-7B-Instruct | Verified |
| Any HuggingFace CausalLM | Works |
| OpenAI API | Verified (92.6% token reduction) |

## Roadmap

- [x] Phase 1 - Python SDK (serialize, client, agent_loop)
- [x] Phase 2 - HTTP API (FastAPI, PacketStore, 43ms/step)
- [x] Phase 3 - float16 blobs (50% smaller), DynamicCache compat
- [x] Phase 4 - Session server (in-memory KV, base dedup, 99% cache hit)
- [x] OpenAI integration (92.6% token reduction, 73.4% cost reduction)
- [x] v0.2 - Stateless inference protocol (hash chain, compact, pure function server)
- [x] Multi-model benchmarks (GPT-2, Qwen2.5-3B, Mistral-7B, OpenAI API)
- [ ] GPU benchmarks (3-4x speedup expected)
- [ ] Auto-compaction heuristics in SDK
- [ ] LangChain/LangGraph native integration
- [ ] Rust HTTP server (protocol layer in Rust, Python inference sidecar)
- [ ] Academic paper (MLSys / arXiv)

## License

MIT