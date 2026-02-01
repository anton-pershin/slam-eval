from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

from slam_eval.utils.typing import HasStr


class Scorer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> int | float: ...


class ExactMatch(Scorer):
    def __init__(
        self,
        name: str,
        preprocessing_func: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(name)
        self.preprocessing_func = preprocessing_func

    def _preprocess(self, value: Any) -> Any:
        if self.preprocessing_func is None:
            return value
        return self.preprocessing_func(value)

    def __call__(self, y_true: Any, y_pred: Any) -> int | float:
        processed_true = self._preprocess(y_true)
        processed_pred = self._preprocess(y_pred)
        return int(processed_true == processed_pred)


def json_string_to_dict(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def build_json_string_to_dict() -> Callable[[Any], Any]:
    return json_string_to_dict


class IgnoreAllWhitespaces(Scorer):
    def __call__(self, y_true: HasStr, y_pred: HasStr) -> int | float:
        # Convert both inputs to strings
        y_true_str = str(y_true)
        y_pred_str = str(y_pred)

        # Create regex pattern from y_true that allows whitespace between characters
        escaped_chars = [re.escape(char) for char in y_true_str if not char.isspace()]
        if not escaped_chars:
            # y_true is whitespace-only, so y_pred must also be whitespace-only
            contains_non_whitespace = any(
                char for char in y_pred_str if not char.isspace()
            )
            return int(not contains_non_whitespace)

        pattern = r"\s*".join(escaped_chars)
        pattern = rf"^\s*{pattern}\s*$"

        # Check if y_pred matches this pattern
        return int(bool(re.match(pattern, y_pred_str)))
