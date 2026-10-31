"""Latency and cost accounting."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

# USD per million tokens (input, output). Local models cost $0 in API spend.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1e6


@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0) * 1000


def latency_summary(values_ms: list[float]) -> dict:
    if not values_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    arr = np.asarray(values_ms)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean())}
