# State Pack

State Pack is a **content-addressed transformer state protocol**.

It treats model state, context, and inference as **verifiable packets**, not sessions.

---

## Core Idea

Instead of:

```
prompt → model → response (ephemeral, opaque)
```

State Pack does:

```
state packet + delta packet → infer → receipt
```

Everything is:

* **content-addressed (SHA-256)**
* **independently verifiable**
* **replayable**

---

## Lifecycle

### 1. CREATE — State Packet

```bash
cargo run -- create --model gpt2 --base demo/base.txt --blob demo/blob.bin --out demo/store
```

Output:

```json
{
  "receipt_id": "sha256:...",
  "op": "create",
  "ok": true,
  "packet_id": "...",
  "base_sha256": "...",
  "blob_sha256": "...",
  "bytes": 1048576
}
```

Creates:

* `state_packet_<hash>.json` (manifest)
* `state_packet_<hash>.pt` (KV cache blob)

---

### 2. VERIFY — Integrity

```bash
cargo run -- verify --manifest demo/store/state_packet_<hash>.json
```

Output:

```json
{
  "receipt_id": "sha256:...",
  "op": "verify",
  "ok": true,
  "packet_id": "...",
  "base_sha256": "...",
  "blob_sha256": "...",
  "bytes": 1048576
}
```

Guarantees:

* blob matches hash
* packet_id is correct
* state is untampered

---

### 3. DELTA — Routing Primitive

```bash
cargo run -- delta \
  --manifest demo/store/state_packet_<hash>.json \
  --delta examples/delta.txt \
  --out demo/delta_packet.json
```

Output:

```json
{
  "receipt_id": "sha256:...",
  "op": "delta",
  "ok": true,
  "packet_id": "...",
  "base_sha256": "...",
  "delta_sha256": "...",
  "bytes": 57
}
```

Delta packet contains:

* pointer to state (`base_sha256`)
* new information only
* no KV cache

---

### 4. INFER — Stateless Execution

```bash
cargo run -- infer \
  --delta demo/delta_packet.json \
  --store demo/store
```

Output:

```json
{
  "receipt_id": "sha256:...",
  "op": "infer",
  "ok": true,
  "packet_id": "...",
  "base_sha256": "...",
  "blob_sha256": "...",
  "delta_sha256": "...",
  "bytes": 57
}
```

Steps:

1. Resolve base state
2. Verify integrity
3. Apply delta
4. Emit receipt

---

### 5. TOKENIZE — Deterministic Token Trace

```bash
cargo run -- tokenize --delta demo/delta_packet.json
```

Output:

```json
{
  "model": "gpt2",
  "delta_sha256": "...",
  "token_count": 15,
  "token_ids": [...],
  "token_trace_sha256": "..."
}
```

This produces a **canonical token sequence** for the delta.

---

## Architecture

```
CREATE → VERIFY → DELTA → INFER → TOKENIZE
```

| Component | Role               |
| --------- | ------------------ |
| base.txt  | semantic input     |
| blob.bin  | KV cache           |
| manifest  | state binding      |
| delta     | new information    |
| receipt   | proof of execution |

---

## Guarantees

* **Content Addressability**

  * All artifacts keyed by SHA-256

* **Deterministic Replay**

  * Same inputs → same outputs

* **Tamper Detection**

  * Any corruption → verify fails

* **State Deduplication**

  * Identical context → identical hash

* **Stateless Inference**

  * No persistent sessions required

---

## Key Insight

This system replaces:

```
persistent conversation state
```

with:

```
portable, verifiable state packets
```

---

## Token Economics

Traditional:

```
cost ∝ total tokens processed
```

State Pack:

```
cost ∝ delta tokens (new information)
```

---

## Repository Structure

```
src/main.rs        CLI + protocol logic
gpt2_tokenize.py   tokenizer bridge
demo/              sample inputs + outputs
examples/          reusable delta/base samples
```

---

## Status

Current version:

```
v0.1 — content-addressed state + delta + infer + token trace
```

Next:

* receipt chaining
* logits trace
* entropy pricing
* distributed packet store

---

## License

MUI
