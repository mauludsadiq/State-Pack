# State Pack

State Pack is a Rust prototype for **content-addressed transformer state packet transport**.

It packages a cached transformer state blob, such as a GPT-2 `past_key_values` `.pt` file, under a deterministic address derived from the base semantic state. A later delta packet can point at that state address and carry only the new text.

This turns full prompt replay into `state_packet(base_hash) + delta_text -> model resumes from cached state`.

The Python GPT-2 experiment already proved the compute primitive:

```text
base_tokens: 931
delta_tokens: 14
full_tokens: 945
full_seconds: 0.646858
packet_delta_seconds: 0.054928
load_seconds: 0.011197
compute_speedup_excluding_load: 11.776x
end_to_end_speedup_including_load: 9.782x
max_abs_logit_diff: 8.392333984375e-05
```

State Pack is the Rust transport layer for that mechanism.

## Core Idea

A state packet has two files:

```text
state_packet_<base_sha256>.pt
state_packet_<base_sha256>.json
```

The base hash is computed from exact base text bytes:

```text
base_sha256 = SHA256(base_text_bytes)
```

The blob hash is computed from raw cached-state bytes:

```text
blob_sha256 = SHA256(state_blob_bytes)
```

The packet ID commits to packet version, model, base hash, base byte length, blob hash, and blob byte length:

```text
packet_id = SHA256(version | model | base_sha256 | base_bytes | blob_sha256 | blob_bytes)
```

That creates a content-addressed object suitable for routing, storage, replay, and verification.

## Build

```bash
cargo build --release
```

## Commands

### Create a state packet

```bash
cargo run -- create \
  --model gpt2 \
  --base examples/base.txt \
  --blob examples/fake_state_packet.pt \
  --out packets
```

Output shape:

```text
created=true
packet_id=sha256:<packet_id>
base_sha256=<base_hash>
blob_sha256=<blob_hash>
manifest=packets/state_packet_<base_hash>.json
blob=packets/state_packet_<base_hash>.pt
```

### Verify a state packet

```bash
cargo run -- verify \
  --manifest packets/state_packet_<base_hash>.json
```

Output shape:

```text
ok=true
packet_id=sha256:<packet_id>
base_sha256=<base_hash>
blob_sha256=<blob_hash>
blob_bytes=<n>
```

### Create a delta packet

```bash
cargo run -- delta \
  --manifest packets/state_packet_<base_hash>.json \
  --delta examples/delta.txt \
  --out packets/delta_packet.json
```

Output shape:

```text
created=true
delta_packet=packets/delta_packet.json
delta_sha256=<delta_hash>
```

### Resolve a state packet by base hash

```bash
cargo run -- resolve \
  --store packets \
  --base-hash <base_hash>
```

Output shape:

```text
manifest=packets/state_packet_<base_hash>.json
blob=packets/state_packet_<base_hash>.pt
```

## Manifest Format

```json
{
  "version": "state-pack-v0.1",
  "model": "gpt2",
  "base_sha256": "...",
  "base_bytes": 178,
  "blob_sha256": "...",
  "blob_bytes": 47,
  "blob_file": "state_packet_<base_sha256>.pt",
  "packet_id": "sha256:..."
}
```

## Delta Packet Format

```json
{
  "version": "state-delta-v0.1",
  "model": "gpt2",
  "base_sha256": "...",
  "packet_id": "sha256:...",
  "delta_sha256": "...",
  "delta_bytes": 55,
  "delta_text": "The next issue is waiver, estoppel, and material breach."
}
```

This represents:

```text
continue_from(packet_id, base_sha256) with delta_text
```

The inference node resolves `base_sha256`, loads the cached state blob, verifies `packet_id`, and runs only the delta tokens.

## GPT-2 Mapping

The earlier Python test did this:

```text
1. Compute GPT-2 KV cache for a 931-token base.
2. Save that cache to disk as state_packet.pt.
3. Reload the cache.
4. Run only a 14-token delta with past_key_values.
5. Compare logits against full 945-token recompute.
```

State Pack formalizes steps 2 and 3:

```text
state_packet.pt
  -> state_packet_<base_sha256>.pt
  -> state_packet_<base_sha256>.json
```

Then the delta packet supplies:

```text
base_sha256 + packet_id + delta_text
```

That is the minimal transport object for semantic packet routing.

## Why This Matters

Current inference is session-resident:

```text
user session -> full context -> VRAM residency
```

State Pack moves toward packetized inference:

```text
semantic address -> cached state packet -> delta-only inference burst
```

Cost shifts from total context length to residual novelty:

```text
cost proportional to H(delta)
```

not total context tokens.

## Next Integration Step

The next layer is a Python/Rust bridge:

```text
Rust State Pack resolves packet
Python loads .pt KV cache
GPT-2 runs delta tokens
Rust records receipt
```

Target receipt:

```json
{
  "model": "gpt2",
  "base_sha256": "...",
  "packet_id": "sha256:...",
  "delta_sha256": "...",
  "output_logits_sha256": "...",
  "max_abs_logit_diff_vs_full": 0.00008392333984375
}
```
