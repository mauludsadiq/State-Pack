"""
state_pack.client
~~~~~~~~~~~~~~~~~
High-level StatePack client. Connects Python inference to the
content-addressed Rust store via subprocess + blob serialization.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch

from .serialize import save_kv_cache, load_kv_cache, BLOB_VERSION

CLI = "cargo run --quiet --"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run(args: list[str]) -> dict:
    """Run state-pack CLI, return parsed receipt."""
    cmd = ["cargo", "run", "--quiet", "--"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"state-pack CLI failed:\n{result.stderr}")
    # CLI may emit warnings before JSON; find first '{'
    out = result.stdout
    idx = out.find("{")
    if idx == -1:
        raise RuntimeError(f"No JSON in CLI output:\n{out}")
    return json.loads(out[idx:])


class StatePack:
    """
    Connects live HuggingFace inference to the State Pack content-addressed store.

    Usage:
        sp = StatePack(store="demo/store", model_id="gpt2")

        # After running base prompt through your model:
        receipt = sp.create(base_text, past_key_values, base_tokens)

        # On each subsequent step:
        receipt = sp.infer(base_sha256, delta_text)

        # Optionally merge accumulated deltas back into base:
        receipt = sp.merge(base_sha256, delta_text, new_past_key_values)
    """

    def __init__(self, store: str | Path = "packets", model_id: str = "gpt2"):
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def create(
        self,
        base_text: str,
        past_key_values,
        base_tokens: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict:
        """
        Serialize KV cache and register it in the store.
        Returns the CLI receipt dict.
        """
        base_sha256 = _sha256(base_text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_path = tmp / "base.txt"
            blob_path = tmp / f"state_packet_{base_sha256}.pt"

            base_path.write_text(base_text, encoding="utf-8")
            save_kv_cache(
                blob_path,
                past_key_values,
                model_id=self.model_id,
                base_text=base_text,
                base_tokens=base_tokens,
                dtype=dtype,
            )

            receipt = _run([
                "create",
                "--model", self.model_id,
                "--base", str(base_path),
                "--blob", str(blob_path),
                "--out", str(self.store),
            ])

        return receipt

    def infer(self, base_sha256: str, delta_text: str) -> dict:
        """
        Verify base state exists and emit an infer receipt for delta_text.
        Does NOT run model inference — call your model separately with
        the KV cache returned by load_base().
        Returns the CLI receipt dict.
        """
        manifest_path = self.store / f"state_packet_{base_sha256}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No state packet for base_sha256={base_sha256}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            delta_path = tmp / "delta.txt"
            delta_packet_path = tmp / "delta_packet.json"

            delta_path.write_text(delta_text, encoding="utf-8")

            _run([
                "delta",
                "--manifest", str(manifest_path),
                "--delta", str(delta_path),
                "--out", str(delta_packet_path),
            ])

            receipt = _run([
                "infer",
                "--delta", str(delta_packet_path),
                "--store", str(self.store),
            ])

        return receipt

    def merge(
        self,
        base_sha256: str,
        delta_text: str,
        new_past_key_values,
        base_tokens_after_merge: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict:
        """
        Merge a delta into the base by saving the new KV cache (post-delta)
        and registering a merged packet in the store.
        Returns the CLI receipt dict.
        """
        manifest_path = self.store / f"state_packet_{base_sha256}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No state packet for base_sha256={base_sha256}")

        merged_preimage = f"{base_sha256}|{_sha256(delta_text)}"
        merged_sha256 = _sha256(merged_preimage)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            delta_path = tmp / "delta.txt"
            delta_packet_path = tmp / "delta_packet.json"
            blob_path = tmp / f"state_packet_{merged_sha256}.pt"

            delta_path.write_text(delta_text, encoding="utf-8")

            # Build merged base text for blob metadata
            merged_base_text = merged_preimage  # hash chain as identity
            save_kv_cache(
                blob_path,
                new_past_key_values,
                model_id=self.model_id,
                base_text=merged_base_text,
                base_tokens=base_tokens_after_merge,
                dtype=dtype,
            )

            _run([
                "delta",
                "--manifest", str(manifest_path),
                "--delta", str(delta_path),
                "--out", str(delta_packet_path),
            ])

            receipt = _run([
                "merge",
                "--manifest", str(manifest_path),
                "--delta", str(delta_packet_path),
                "--blob", str(blob_path),
                "--out", str(self.store),
            ])

        return receipt

    def load_base(self, base_sha256: str, map_location: str = "cpu") -> dict:
        """
        Load the KV cache blob for a given base_sha256.
        Returns the full blob dict including past_key_values.
        """
        blob_path = self.store / f"state_packet_{base_sha256}.pt"
        if not blob_path.exists():
            raise FileNotFoundError(f"No blob for base_sha256={base_sha256}")
        return load_kv_cache(blob_path, map_location=map_location)

    def verify(self, base_sha256: str) -> dict:
        """Verify integrity of a stored packet. Returns CLI receipt."""
        manifest_path = self.store / f"state_packet_{base_sha256}.json"
        return _run(["verify", "--manifest", str(manifest_path)])

    def resolve(self, base_sha256: str) -> dict:
        """Return paths to manifest and blob for a given hash."""
        manifest = self.store / f"state_packet_{base_sha256}.json"
        blob = self.store / f"state_packet_{base_sha256}.pt"
        if not manifest.exists():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        if not blob.exists():
            raise FileNotFoundError(f"blob not found: {blob}")
        return {"manifest": str(manifest), "blob": str(blob)}
