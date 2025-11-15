from typing import Any, Protocol


class Scorer(Protocol):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(y_true: Any, y_pred: Any) -> int | float:
        ...


class ExactMatch(Scorer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, y_true: Any, y_pred: Any) -> int | float:
        return int(y_true == y_pred)

