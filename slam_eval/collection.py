from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypedDict, Optional

import datasets


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
        self.collection: Optional[Any] = None

    def __iter__(self) -> EvalCaseCollection:
        return self

    @abstractmethod
    def load(self) -> None:
        """This method populates self.collection"""
        ...

    @abstractmethod
    def __next__(self) -> EvalCase:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class HuggingFaceCollection(EvalCaseCollection):
    def __init__(
        self,
        name: str,
        dataset_name: str,
        split: Optional[str] = None,
        subset: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.collection_iter = None

    def load(self) -> None:
        self.collection = datasets.load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split
        )
        self.collection_iter = iter(self.collection)

    @check_if_loaded
    def __next__(self) -> EvalCase:
        raw_item = next(self.collection_iter)
        return EvalCase(x=raw_item["input"], y_true=raw_item["target"])
        
    @check_if_loaded
    def __len__(self) -> int:
        return self.collection.num_rows


class CollectionNotLoadedError(Exception):
    pass
