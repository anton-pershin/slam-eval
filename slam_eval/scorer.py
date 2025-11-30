from abc import ABC, abstractmethod
from typing import Any


class Scorer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> int | float: ...


class ExactMatch(Scorer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, y_true: Any, y_pred: Any) -> int | float:
        return int(y_true == y_pred)
