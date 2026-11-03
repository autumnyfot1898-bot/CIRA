"""Append-only progress records: pre-action inputs are durable before execution."""
from __future__ import annotations
import os
import threading
import time
from pathlib import Path
from .state import canonical


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def emit(self, event, **fields):
        payload = canonical({"event": event, "time": time.time(), **fields}) + b"\n"
        with self.lock, open(self.path, "ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())


class DurableRecords(list):
    def __init__(self, path):
        import json
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        values = [json.loads(line) for line in self.path.read_bytes().splitlines()] if self.path.is_file() else []
        super().__init__(values)

    def append(self, value):
        with self.lock, open(self.path, "ab") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            super().append(value)
