from __future__ import annotations

from typing import Any, Callable


class IFBenchCheckerFactory:
    def __init__(self, registry: dict[str, Callable[[], Any]] | None = None) -> None:
        self._registry = registry or {}

    def register(self, instruction_id: str, factory: Callable[[], Any]) -> None:
        self._registry[instruction_id] = factory

    def __call__(self, instruction_id: str) -> Any:
        if instruction_id not in self._registry:
            raise KeyError(f"Unknown IFBench instruction id: {instruction_id}")
        return self._registry[instruction_id](instruction_id)
