from __future__ import annotations
from transformers.cache_utils import DynamicCache
"""
state_pack.store
~~~~~~~~~~~~~~~~
Pure Python implementation of the State Pack store protocol.
No subprocess. Same receipt format as the Rust CLI.
Used by the server for low-latency infer/delta/merge operations.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import torch

from .serialize import save_kv_cache, load_kv_cache, BLOB_VERSION

PACKET_VERSION = "state-pack-v0.1"


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _receipt(op: str, **kwargs) -> dict:
    r = {"receipt_id": None, "op": op, "ok": True, **kwargs}
    canonical = json.dumps({k: v for k, v in r.items() if k != "receipt_id"},
                           sort_keys=True, separators=(",", ":")).encode()
    r["receipt_id"] = "sha256:" + _sha256(canonical)
    return r


class PacketStore:
    """
    In-process State Pack store. Same semantics as the Rust CLI,
    no subprocess overhead.
    """

    def __init__(self, store: str | Path, model_id: str = "gpt2"):
        self.store    = Path(store)
        self.model_id = model_id
        self.store.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def create(self, base_text: str, past_key_values, base_tokens: int,
               dtype: Optional[torch.dtype] = None) -> dict:
        base_sha256 = _sha256(base_text)
        blob_path   = self.store / f"state_packet_{base_sha256}.pt"

        meta = save_kv_cache(
            blob_path, past_key_values,
            model_id=self.model_id,
            base_text=base_text,
            base_tokens=base_tokens,
            dtype=dtype,
        )

        manifest = {
            "version":     PACKET_VERSION,
            "model":       self.model_id,
            "base_sha256": base_sha256,
            "base_bytes":  len(base_text.encode()),
            "base_tokens": base_tokens,
            "base_file":   Path(blob_path).name,
            "blob_sha256": meta["base_sha256"],  # blob self-identifies
            "blob_bytes":  meta["bytes"],
            "blob_file":   blob_path.name,
            "packet_id":   "",
        }
        # recompute blob_sha256 from file
        manifest["blob_sha256"] = _sha256(blob_path.read_bytes())
        manifest["blob_bytes"]  = blob_path.stat().st_size

        preimage = "|".join([
            manifest["version"], manifest["model"],
            manifest["base_sha256"], str(manifest["base_bytes"]),
            manifest["blob_sha256"], str(manifest["blob_bytes"]),
        ])
        manifest["packet_id"] = "sha256:" + _sha256(preimage)

        mpath = self.store / f"state_packet_{base_sha256}.json"
        mpath.write_text(json.dumps(manifest, indent=2))

        return _receipt(
            "create",
            packet_id=manifest["packet_id"],
            base_sha256=base_sha256,
            blob_sha256=manifest["blob_sha256"],
            bytes=manifest["blob_bytes"],
        )

    def infer(self, base_sha256: str, delta_text: str,
              delta_tokens: int, base_tokens: int) -> dict:
        """
        Emit an infer receipt. Caller is responsible for running the model.
        delta_tokens / base_tokens must be provided by the caller.
        """
        manifest = self._load_manifest(base_sha256)
        delta_sha256 = _sha256(delta_text)
        delta_bytes  = len(delta_text.encode())
        base_bytes   = manifest["base_bytes"]

        return _receipt(
            "infer",
            packet_id=manifest["packet_id"],
            base_sha256=base_sha256,
            blob_sha256=manifest["blob_sha256"],
            delta_sha256=delta_sha256,
            bytes=delta_bytes,
            bytes_saved={
                "base": base_bytes,
                "delta": delta_bytes,
                "processed": delta_bytes,
                "saved": base_bytes,
                "savings_percent": base_bytes / (base_bytes + delta_bytes) * 100
                    if (base_bytes + delta_bytes) else 0,
            },
            tokens={
                "base": base_tokens,
                "delta": delta_tokens,
                "processed": delta_tokens,
                "saved": base_tokens,
                "savings_percent": base_tokens / (base_tokens + delta_tokens) * 100
                    if (base_tokens + delta_tokens) else 0,
            },
        )

    def merge(self, base_sha256: str, delta_text: str,
              new_past_key_values, new_base_tokens: int,
              dtype: Optional[torch.dtype] = None) -> dict:
        manifest    = self._load_manifest(base_sha256)
        delta_sha256 = _sha256(delta_text)

        merged_sha256 = _sha256(f"{base_sha256}|{delta_sha256}")
        blob_path     = self.store / f"state_packet_{merged_sha256}.pt"
        merged_text   = f"{base_sha256}|{delta_sha256}"

        meta = save_kv_cache(
            blob_path, new_past_key_values,
            model_id=self.model_id,
            base_text=merged_text,
            base_tokens=new_base_tokens,
            dtype=dtype,
        )

        blob_sha256 = _sha256(blob_path.read_bytes())
        blob_bytes  = blob_path.stat().st_size

        merged = {
            "version":     PACKET_VERSION,
            "model":       self.model_id,
            "base_sha256": merged_sha256,
            "base_bytes":  manifest["base_bytes"] + len(delta_text.encode()),
            "base_tokens": new_base_tokens,
            "base_file":   f"merge:{base_sha256}+{delta_sha256}",
            "blob_sha256": blob_sha256,
            "blob_bytes":  blob_bytes,
            "blob_file":   blob_path.name,
            "packet_id":   "",
        }
        preimage = "|".join([
            merged["version"], merged["model"],
            merged["base_sha256"], str(merged["base_bytes"]),
            merged["blob_sha256"], str(merged["blob_bytes"]),
        ])
        merged["packet_id"] = "sha256:" + _sha256(preimage)

        mpath = self.store / f"state_packet_{merged_sha256}.json"
        mpath.write_text(json.dumps(merged, indent=2))

        return _receipt(
            "merge",
            packet_id=merged["packet_id"],
            base_sha256=merged_sha256,
            blob_sha256=blob_sha256,
            delta_sha256=delta_sha256,
            bytes=blob_bytes,
        )

    def load_kv(self, base_sha256: str, map_location: str = "cpu") -> dict:
        blob_path = self.store / f"state_packet_{base_sha256}.pt"
        if not blob_path.exists():
            raise FileNotFoundError(f"No blob for {base_sha256}")
        return load_kv_cache(blob_path, map_location=map_location)

    def verify(self, base_sha256: str) -> dict:
        manifest  = self._load_manifest(base_sha256)
        blob_path = self.store / manifest["blob_file"]
        actual    = _sha256(blob_path.read_bytes())
        if actual != manifest["blob_sha256"]:
            raise ValueError(f"blob hash mismatch: {actual} != {manifest['blob_sha256']}")
        return _receipt(
            "verify",
            packet_id=manifest["packet_id"],
            base_sha256=base_sha256,
            blob_sha256=manifest["blob_sha256"],
            bytes=manifest["blob_bytes"],
        )

    def _load_manifest(self, base_sha256: str) -> dict:
        mpath = self.store / f"state_packet_{base_sha256}.json"
        if not mpath.exists():
            raise FileNotFoundError(f"No manifest for {base_sha256}")
        return json.loads(mpath.read_text())


# Default dtype for new blobs — float16 halves size with no quality loss
DEFAULT_BLOB_DTYPE = torch.float16