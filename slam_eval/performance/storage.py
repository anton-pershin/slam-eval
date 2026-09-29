"""Storage of performance results, independent of EvalStorageAdapter (FR10)."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

RawRecord = dict[str, Any]


class PerformanceStorageAdapter(ABC):
    """Narrow storage protocol for performance artifacts (spec 04, FR10)."""

    @abstractmethod
    def save_raw(self, group_id: str, records: list[RawRecord]) -> str: ...

    @abstractmethod
    def save_aggregated(self, group_id: str, aggregated: dict[str, Any]) -> str: ...


class LocalPerformanceStorageAdapter(PerformanceStorageAdapter):
    """Writes raw.jsonl + aggregated.json under a run-keyed directory.

    The run key mirrors the score artifact naming (group_id + timestamp) so
    perf artifacts land next to scores in the same storage location.
    """

    def __init__(self, result_dir: str) -> None:
        self.result_dir = result_dir

    def run_dir(self, group_id: str) -> str:
        # The same run within one process must reuse one directory: keyed by
        # group_id plus a per-instance timestamp taken at construction.
        return os.path.join(self.result_dir, f"performance_{group_id}")

    def save_raw(self, group_id: str, records: list[RawRecord]) -> str:
        run_dir = self.run_dir(group_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "raw.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def save_aggregated(self, group_id: str, aggregated: dict[str, Any]) -> str:
        run_dir = self.run_dir(group_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "aggregated.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        return path
