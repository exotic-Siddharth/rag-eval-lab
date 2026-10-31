"""Thread-safe SQLite key-value cache for embeddings and LLM responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class KVCache:
    def __init__(self, path: str | Path = ".cache/raglab.sqlite"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, value BLOB, PRIMARY KEY (ns, key))")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_many(self, ns: str, keys: list[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                batch = keys[i : i + 500]
                marks = ",".join("?" * len(batch))
                rows = self._db.execute(f"SELECT key, value FROM kv WHERE ns=? AND key IN ({marks})", [ns, *batch])
                found.update(rows.fetchall())
        self.hits += len(found)
        self.misses += len(keys) - len(found)
        return found

    def put_many(self, ns: str, items: dict[str, bytes]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", [(ns, k, v) for k, v in items.items()])
            self._db.commit()

    def get_json(self, ns: str, key: str):
        row = self.get_many(ns, [key]).get(key)
        return None if row is None else json.loads(row)

    def put_json(self, ns: str, key: str, value) -> None:
        self.put_many(ns, {key: json.dumps(value).encode("utf-8")})

    def get_vectors(self, ns: str, keys: list[str]) -> dict[str, np.ndarray]:
        return {k: np.frombuffer(v, dtype=np.float32) for k, v in self.get_many(ns, keys).items()}

    def put_vectors(self, ns: str, items: dict[str, np.ndarray]) -> None:
        self.put_many(ns, {k: np.asarray(v, dtype=np.float32).tobytes() for k, v in items.items()})
