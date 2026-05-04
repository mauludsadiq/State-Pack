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

All benchmarks run over 20-step agent loops.
GPU inference is expected to show 3-4x wall-clock improvement over the CPU numbers above.

## Cost Impact

| | Naive | State Pack | Saving |
|--|-------|------------|--------|
| Input tokens (20-step loop) | 17,929 | 1,320 | 92.6% |
| Cost per loop — gpt-4o-mini | $0.00341 | $0.00091 | 73.4% |
| Cost per loop — gpt-4o | $0.180 | $0.048 | 73.4% |
| 1,000 agents x 100 loops/day — gpt-4o | $14,440 | $3,632 | $10,808/day |

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
Compaction is triggered automatically by configurable policy, or manually by the client.

## On the OpenAI Integration

The OpenAI benchmark does not transfer local KV cache tensors to OpenAI's servers —
that API surface does not exist. Instead, State Pack achieves savings through
structured context discipline: only the system prompt and the current delta are sent
each step, rather than the full growing conversation history.

This is a different mechanism from local inference but produces the same structural
savings. OpenAI's own prompt caching may additionally cache the repeated system
prompt prefix, compounding the reduction. The 92.6% figure is real and reproducible
on your own key.

## Architecture (v0.3)

v0.3 introduces a split architecture: the Rust server owns the protocol layer
(content addressing, receipts, store I/O) and Python owns inference.
This eliminates GIL contention on the hot path and brings protocol latency
from 107ms to 9.1ms — a 12x improvement.

```
                   +------------------+
   agent loop ---> | Python inference |  47ms  (model bound)
                   +------------------+
                            |
                            v
                   +------------------+
                   |  Rust server     |  9ms   (protocol, receipts, store)
                   +------------------+
                            |
                            v
                   +------------------+
                   |  blob store      |  SHA-256 addressed .pt files
                   +------------------+
```

## The Stateless Protocol

The server is a pure function. Zero session state. The client owns the hash chain.

```
POST /states        { base_text }          -> { state_hash }
POST /infer         { state_hash, delta }   -> { new_state_hash, output, savings }
POST /merge         { state_hash, delta }   -> { new_state_hash }
POST /compact       { state_hash, deltas }  -> { new_state_hash, steps_folded }
GET  /states/{hash}                         -> { tokens, bytes, hot }
GET  /health                                -> { states_cached, requests_served }
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

### Run the full stack (Rust protocol + Python inference)

```bash
pip install state-pack

# Start Rust protocol server
cargo build --bin state-pack-server --release
./target/release/state-pack-server --store my_store --model gpt2 --port 8003

# Run agent loop with auto-compaction
python3 - <<'EOF'
from state_pack.stateless_client import StatelessClient
from state_pack.compaction import ThresholdPolicy

client = StatelessClient(
    model_id='gpt2',
    rust_server='http://localhost:8003',
    store='my_store',
    policy=ThresholdPolicy(token_ratio=1.0, max_steps=20),
)

client.set_base('You are a research agent...\n\n')

for delta in steps:
    result = client.step(delta)
    print(result['savings_pct'], result['compacted'])

print(client.stats())
EOF
```

### Python-only (no Rust server required)

```bash
PYTHONPATH=. python3 -m state_pack.stateless_server --store my_store --model gpt2
```

### OpenAI API (no local model required)

```bash
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python3 examples/openai_benchmark.py
```

## Compaction Policies

```python
from state_pack.compaction import ThresholdPolicy, SavingsPolicy, NeverPolicy

# Compact when delta tokens exceed base tokens (recommended default)
ThresholdPolicy(token_ratio=1.0, max_steps=20, min_steps=3)

# Compact when per-step savings drop below 70%
SavingsPolicy(min_savings_pct=70.0, max_steps=30)

# Never compact (manual control)
NeverPolicy()
```

## File Reference

```
src/
  main.rs                Rust CLI - content addressing, receipts, benchmarks
  server.rs              Rust HTTP server - protocol layer, 9ms latency

state_pack/
  stateless_client.py    Production agent loop client (Rust + Python)
  stateless_server.py    Python-only stateless server
  compaction.py          Auto-compaction policies
  session_server.py      In-memory session server - base deduplication
  server.py              Simple HTTP API
  llm.py                 Drop-in LLM wrapper
  store.py               In-process packet store
  serialize.py           KV cache serialization - float16
  client.py              High-level SDK
  agent_loop.py          Agent loop benchmark
  openai_integration.py  OpenAI API benchmark

examples/
  openai_benchmark.py    OpenAI cost benchmark
  sdk_benchmark.py       Local inference benchmark
  mistral_benchmark.py   Mistral-7B benchmark
  qwen_benchmark.py      Qwen2.5-3B benchmark

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

- [x] Python SDK
- [x] HTTP API - FastAPI, 43ms/step
- [x] float16 blobs - 50% smaller, zero quality loss
- [x] Session server - in-memory KV, base deduplication
- [x] OpenAI integration - 92.6% token reduction
- [x] Stateless protocol - pure function server, client-owned hash chain
- [x] Multi-model benchmarks - GPT-2, Qwen2.5-3B, Mistral-7B, OpenAI
- [x] Rust HTTP server - 9.1ms protocol latency, 12x faster than Python
- [x] Auto-compaction - ThresholdPolicy, SavingsPolicy, configurable
- [x] Interactive savings calculator
- [ ] GPU benchmarks and optimized KV transfer
- [ ] LangChain / LangGraph integration
- [ ] Academic paper (MLSys / arXiv)

## License

MIT