"""
state_pack.serialize
~~~~~~~~~~~~~~~~~~~~
Serialize/deserialize transformer past_key_values to/from .pt blobs.

Blob format (torch.save dict):
    {
        "state_pack_version": "state-pack-v0.1",
        "model":              str,
        "base_sha256":        str,
        "base_tokens":        int,
        "past_key_values":    tuple,
        "dtype":              str,
        "device":             str,
        "num_layers":         int,
        "seq_len":            int,
    }
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch

BLOB_VERSION = "state-pack-v0.1"
from transformers.cache_utils import DynamicCache


def to_dynamic_cache(past_key_values):
    if hasattr(DynamicCache, 'from_legacy_cache'):
        return DynamicCache.from_legacy_cache(past_key_values)
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_key_values):
        cache.update(k, v, layer_idx)
    return cache


def from_dynamic_cache(cache):
    if hasattr(cache, 'to_legacy_cache'):
        return cache.to_legacy_cache()
    if hasattr(cache, 'key_cache'):
        return tuple((cache.key_cache[i], cache.value_cache[i]) for i in range(len(cache.key_cache)))
    return tuple(cache)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _infer_meta(past_key_values) -> dict:
    layer = past_key_values[0]
    k = layer[0]
    return {
        "num_layers": len(past_key_values),
        "seq_len":    k.shape[-2],
        "dtype":      str(k.dtype).replace("torch.", ""),
        "device":     str(k.device),
    }


def save_kv_cache(
    path: str | Path,
    past_key_values,
    model_id: str,
    base_text: str,
    base_tokens: int,
    dtype: Optional[torch.dtype] = None,
) -> dict:
    """
    Save a HuggingFace past_key_values tuple as a State Pack blob.
    Returns metadata dict (no tensors) for building a manifest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        past_key_values = tuple(
            tuple(t.to(dtype) for t in layer)
            for layer in past_key_values
        )

    meta = _infer_meta(past_key_values)
    base_sha256 = _sha256(base_text)

    blob = {
        "state_pack_version": BLOB_VERSION,
        "model":              model_id,
        "base_sha256":        base_sha256,
        "base_tokens":        base_tokens,
        "past_key_values":    past_key_values,
        **meta,
    }

    torch.save(blob, path)

    return {
        "state_pack_version": BLOB_VERSION,
        "model":              model_id,
        "base_sha256":        base_sha256,
        "base_tokens":        base_tokens,
        **meta,
        "path":               str(path),
        "bytes":              path.stat().st_size,
    }


def load_kv_cache(path: str | Path, map_location: str = "cpu") -> dict:
    """
    Load a State Pack blob. Returns full dict including past_key_values.
    Raises ValueError on version mismatch.
    """
    path = Path(path)
    blob = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(blob, dict):
        raise ValueError(f"Expected dict blob, got {type(blob)}")

    ver = blob.get("state_pack_version")
    if ver != BLOB_VERSION:
        raise ValueError(f"Unsupported blob version: {ver!r} (expected {BLOB_VERSION!r})")

    return blob


def kv_cache_meta(path: str | Path, map_location: str = "cpu") -> dict:
    """
    Load blob metadata only (no past_key_values returned).
    Useful for manifest building without holding tensors in memory.
    """
    blob = load_kv_cache(path, map_location=map_location)
    return {k: v for k, v in blob.items() if k != "past_key_values"}
