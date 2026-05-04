from __future__ import annotations
from abc import ABC, abstractmethod

class CompactionPolicy(ABC):
    @abstractmethod
    def should_compact(self, base_tokens: int, accumulated_deltas: list, step: int = 0) -> bool: ...
    def reset(self): pass

class ThresholdPolicy(CompactionPolicy):
    def __init__(self, token_ratio: float = 1.0, max_steps: int = 20, min_steps: int = 3):
        self.token_ratio = token_ratio
        self.max_steps   = max_steps
        self.min_steps   = min_steps
    def should_compact(self, base_tokens, accumulated_deltas, step=0):
        n = len(accumulated_deltas)
        if n < self.min_steps: return False
        if n >= self.max_steps: return True
        delta_tokens = sum(len(d.split()) for d in accumulated_deltas)
        return delta_tokens > base_tokens * self.token_ratio

class SavingsPolicy(CompactionPolicy):
    def __init__(self, min_savings_pct: float = 70.0, max_steps: int = 30):
        self.min_savings_pct = min_savings_pct
        self.max_steps       = max_steps
    def should_compact(self, base_tokens, accumulated_deltas, step=0):
        n = len(accumulated_deltas)
        if n == 0: return False
        if n >= self.max_steps: return True
        avg_delta = sum(len(d.split()) for d in accumulated_deltas) / n
        savings   = base_tokens / (base_tokens + avg_delta) * 100
        return savings < self.min_savings_pct

class NeverPolicy(CompactionPolicy):
    def should_compact(self, base_tokens, accumulated_deltas, step=0): return False

class AlwaysPolicy(CompactionPolicy):
    def should_compact(self, base_tokens, accumulated_deltas, step=0): return len(accumulated_deltas) > 0

DEFAULT_POLICY = ThresholdPolicy(token_ratio=1.0, max_steps=20, min_steps=3)
