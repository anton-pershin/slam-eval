from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from slam_eval.collections.base import EvalCaseCollection


def check_if_loaded(func):
    def wrapper(self, *args, **kwargs):
        if self.collection is None:
            raise CollectionNotLoadedError(
                f"Collection {self.name} not loaded. "
                "Perhaps, you forgot to call load()"
            )
        return func(self, *args, **kwargs)

    return wrapper


class EvalCase(TypedDict):
    x: Any
    y_true: Any


class EvalCaseCollection(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.collection: Optional[Iterable[Any]] = None
        self.collection_len: Optional[int] = None

    def __iter__(self) -> "EvalCaseCollection":
        return self

    def load(self) -> None:
        collection_info = self._load()
        self.collection = collection_info["collection"]
        self.collection_len = collection_info["collection_len"]

    def __len__(self) -> int:
        if self.collection_len is None:
            raise TypeError("Collection not loaded. Call load() first.")
        return self.collection_len

    @abstractmethod
    def _load(self) -> "CollectionInfo": ...

    @abstractmethod
    def __next__(self) -> EvalCase: ...


class CollectionInfo(TypedDict):
    collection: Iterable[Any]
    collection_len: int


class CollectionNotLoadedError(Exception):
    pass
