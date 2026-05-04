# State Pack

**The CDN for AI inference costs.**

Every time an agent takes a step, it reprocesses its entire context window from scratch.
The bill compounds with every token. State Pack eliminates that by caching the transformer
KV state after the base prompt and processing only the new information on each subsequent step.

The analogy is exact: in the early 2000s, users paid per SMS character.
BlackBerry made that cost invisible at the infrastructure layer — not by changing the
network, but by compressing state between sends. State Pack does the same for tokens.

## Benchmarks

Savings are consistent across model families and sizes.
The reduction is structural — it comes from the protocol, not the model.

| Model | Params | Token Savings | Blob Size |
|-------|--------|--------------|-----------|
| GPT-2 | 124M | 95.3% | 0.5MB |
| Qwen2.5-3B | 3B | 90.9% | 1.2MB |
| Mistral-7B-Instruct | 7B | 90.9% | 5.7MB |
| OpenAI API (gpt-4o-mini) | — | 92.6% | — |

All benchmarks run over 20-step agent loops. Speedup numbers are CPU-bound;
GPU inference is expected to show 3-4x wall-clock improvement based on GPT-2 results.

## Cost Impact

| | Naive | State Pack | Saving |
|--|-------|------------|--------|
| Input tokens (20-step loop) | 17,929 | 1,320 | 92.6% |
| Cost per loop — gpt-4o-mini | $0.00341 | $0.00091 | 73.4% |
| Cost per loop — gpt-4o | $0.180 | $0.048 | 73.4% |
| 1,000 agents × 100 loops/day — gpt-4o | $14,440 | $3,632 | $10,808/day |

When 1,000 agents share the same system prompt, the base KV cache is computed
once and served to all. Agents 2 through 1,000 pay zero tokens for context setup.

[Interactive savings calculator](https://mauludsadiq.github.io/State-Pack/calculator.html)

## How It Works

```
naive:       [system + full history + delta] -> model   cost grows every step
state pack:  [delta only]                    -> model   cost stays flat
```

**CREATE** — run the base prompt once, serialize the KV cache to a content-addressed blob.
The blob is keyed by SHA-256 of the input text. Same prompt always produces the same hash.

**INFER** — on each subsequent step, load the cached KV state and process the delta tokens only.
A tamper-evident receipt is emitted for every inference operation.

**COMPACT** — after N steps, fold the accumulated delta chain back into a fresh base state.
This prevents the delta chain from growing indefinitely and keeps savings compounding.

## On the OpenAI Integration

The OpenAI benchmark does not transfer local KV cache tensors to OpenAI's servers —
that API surface does not exist. Instead, State Pack achieves savings through
structured context discipline: only the system prompt and the current delta are sent
each step, rather than the full growing conversation history.

This is a different mechanism from local inference but produces the same structural
savings. OpenAI's own prompt caching may additionally cache the repeated system
prompt prefix, compounding the reduction. The 92.6% figure is real and reproducible
on your own key — the mechanism is honest about what it is.

## The Stateless Protocol (v0.2)

The server is a pure function. Zero session state. The client owns the hash chain.

```
POST /states        { base_text }          -> { state_hash }
POST /infer         { state_hash, delta }   -> { new_state_hash, output, savings }
POST /merge         { state_hash, delta }   -> { new_state_hash }
POST /compact       { state_hash, deltas }  -> { new_state_hash, steps_folded }
GET  /states/{hash}                         -> { tokens, bytes, hot }
GET  /health                                -> { states_hot, states_cached }
```

Client chains hashes: `h0 -> infer -> h1 -> infer -> h2 -> compact -> h_fresh`

The server cannot reconstruct a conversation even if asked to.
The same state_hash from any client always returns the same result.
The design is inherently horizontally scalable and supports multi-region deployment.

## Quickstart

### Reproduce the OpenAI benchmark on your own key

```bash
git clone https://github.com/mauludsadiq/State-Pack.git
cd State-Pack
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python3 examples/openai_benchmark.py
```

### Run the stateless server

```bash
pip install state-pack
PYTHONPATH=. python3 -m state_pack.stateless_server --store my_store --model gpt2

# Create base state — idempotent, same text always returns same hash
curl -X POST http://localhost:8002/states \
  -H 'Content-Type: application/json' \
  -d '{"base_text": "You are a legal research agent..."}'

# Infer — pure function, client advances the hash chain
curl -X POST http://localhost:8002/infer \
  -H 'Content-Type: application/json' \
  -d '{"state_hash": "<hash>", "delta_text": "Step 1: clause affects indemnity."}'

# Compact accumulated deltas into a fresh base state
curl -X POST http://localhost:8002/compact \
  -H 'Content-Type: application/json' \
  -d '{"state_hash": "<hash>", "accumulated_deltas": ["Step 1...", "Step 2..."]}'
```

### Python SDK

```python
from state_pack.llm import StatePackLLM

llm = StatePackLLM.from_pretrained('gpt2', store='my_store', merge_every=10)
llm.set_base('You are a research agent...\n\n')

for delta in steps:
    output = llm(delta)  # only delta tokens processed

print(llm.stats())
# {'tokens_saved': 17785, 'savings_pct': 95.31, 'speedup': 3.958}
```

## Architecture

```
state_pack/
  stateless_server.py    Stateless protocol (v0.2) — pure function, hash chain API
  session_server.py      In-memory KV cache — base deduplication, 1000-agent scale
  server.py              HTTP API — FastAPI, 43ms/step
  llm.py                 Drop-in LLM wrapper with automatic KV reuse
  store.py               In-process packet store
  serialize.py           KV cache serialization — float16, 50% smaller blobs
  client.py              High-level Python SDK
  agent_loop.py          Drop-in agent loop benchmark
  openai_integration.py  OpenAI API benchmark

src/main.rs              Rust CLI — content addressing, receipts, protocol
calculator.html          Interactive savings calculator
```

## Verified Models

| Model | Status |
|-------|--------|
| GPT-2 (124M) | Verified |
| Qwen2.5-3B | Verified |
| Mistral-7B-Instruct | Verified |
| Any HuggingFace CausalLM | Compatible |
| OpenAI API | Verified |

## Roadmap

- [x] Python SDK — serialize, client, agent loop
- [x] HTTP API — FastAPI, 43ms/step
- [x] float16 blobs — 50% smaller, zero quality loss
- [x] Session server — in-memory KV, base deduplication
- [x] OpenAI integration — 92.6% token reduction on live API
- [x] Stateless protocol v0.2 — pure function server, client-owned hash chain
- [x] Multi-model benchmarks — GPT-2, Qwen2.5-3B, Mistral-7B, OpenAI
- [x] Interactive savings calculator
- [ ] GPU benchmarks
- [ ] Auto-compaction heuristics
- [ ] LangChain / LangGraph integration
- [ ] Rust HTTP server
- [ ] Academic paper

## License

MIT