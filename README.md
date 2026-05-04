# State Pack

**The CDN for AI inference costs.**

Agents pay per token. State Pack makes that cost invisible — the same way
BlackBerry made per-character SMS costs invisible. Not by changing the model.
Not by changing the API. By caching state at the infrastructure layer.

## Proven on the OpenAI API

| Metric | Naive | State Pack | Saving |
|--------|-------|------------|--------|
| Input tokens (20-step agent loop) | 18,990 | 1,320 | 93% |
| Cost per loop (gpt-4o-mini) | $0.00361 | $0.00091 | 74% |
| Cost per loop (gpt-4o) | ~$0.190 | ~$0.048 | 74% |
| Latency per step | — | 44ms | — |
| Base cache hit (shared agents) | 0.951s | 0.003s | 99% |

Real numbers. Real API. Real dollars.

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
3. MERGE   - fold deltas back into base on threshold (keeps savings compounding)

Every artifact is SHA-256 addressed. Every operation emits a tamper-evident receipt.
Same inputs always produce same outputs. Fully auditable.

## Prove It Against Your Own API Key

```bash
git clone https://github.com/mauludsadiq/State-Pack.git
cd State-Pack
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python3 examples/openai_benchmark.py
```

## Session Server: 1,000 Agents, 1 Base

```bash
PYTHONPATH=. python3 -m state_pack.session_server --store my_store --model gpt2

# Agent 1 - computes base (0.951s)
curl -X POST http://localhost:8001/sessions \
  -H 'Content-Type: application/json' \
  -d '{"base_text": "You are a legal research agent..."}'

# Agent 2 - cache hit (0.003s)
curl -X POST http://localhost:8001/sessions \
  -H 'Content-Type: application/json' \
  -d '{"base_text": "You are a legal research agent..."}'

# Run a step
curl -X POST http://localhost:8001/sessions/{id}/step \
  -H 'Content-Type: application/json' \
  -d '{"delta_text": "Step 1: clause affects indemnity."}'
```

## Python SDK

```python
from state_pack.llm import StatePackLLM

llm = StatePackLLM.from_pretrained('gpt2', store='my_store', merge_every=10)
llm.set_base('You are a research agent...\n\n')

for delta in steps:
    output = llm(delta)   # only delta tokens processed

print(llm.stats())
# tokens_saved: 17785, savings_pct: 95.31, speedup: 3.958
```

## HTTP API

```bash
PYTHONPATH=. python3 -m state_pack.server --store my_store --model gpt2

curl -X POST http://localhost:8000/packets \
  -H 'Content-Type: application/json' \
  -d '{"base_text": "You are a research agent..."}'

curl -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{"base_sha256": "<sha>", "delta_text": "Step 1."}'
```

## Architecture

```
state_pack/
  session_server.py  In-memory KV cache, base dedup, 1000-agent scale
  server.py          HTTP API (FastAPI, 43ms/step)
  llm.py             Drop-in LLM wrapper with automatic KV reuse
  store.py           In-process packet store (no subprocess)
  serialize.py       KV cache to .pt blob (float16, 50% smaller)
  client.py          High-level SDK
  agent_loop.py      Drop-in agent loop
  openai_integration.py  Benchmark against OpenAI API

src/main.rs          Rust CLI - content addressing, receipts, protocol
```

## Model Support

| Model | Status |
|-------|--------|
| GPT-2 | Verified |
| Llama (tiny) | Verified |
| Any HuggingFace CausalLM | Works |
| OpenAI API | Verified (93% token reduction) |

## Roadmap

- [x] Phase 1 - Python SDK (serialize, client, agent_loop)
- [x] Phase 2 - HTTP API (FastAPI, PacketStore, 43ms/step)
- [x] Phase 3 - float16 blobs (50% smaller), DynamicCache compat
- [x] Phase 4 - Session server (in-memory KV, base dedup, 99% cache hit)
- [x] OpenAI integration (93% token reduction, 74% cost reduction, live API)
- [ ] GPU/multi-device KV portability
- [ ] LangChain/LangGraph native integration
- [ ] Rust HTTP server (protocol layer in Rust, Python inference sidecar)

## License

MUI
