import re
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


class IgnoreAllWhitespaces(Scorer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, y_true: Any, y_pred: Any) -> int | float:
        # Convert both inputs to strings
        y_true_str = str(y_true)
        y_pred_str = str(y_pred)
        
        # Create regex pattern from y_true that allows whitespace between any characters
        # Escape special regex characters and insert \s* between each character
        escaped_chars = [re.escape(char) for char in y_true_str if not char.isspace()]
        if not escaped_chars:
            # If y_true has no non-whitespace characters, check if y_pred is also whitespace-only
            return int(not any(char for char in y_pred_str if not char.isspace()))
        
        pattern = r'\s*'.join(escaped_chars)
        pattern = r'^\s*' + pattern + r'\s*$'
        
        # Check if y_pred matches this pattern
        return int(bool(re.match(pattern, y_pred_str)))
