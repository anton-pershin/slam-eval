from typing import Optional, TypedDict

import datasets

from slam_eval.collections.base import (CollectionInfo, EvalCaseCollection,
                                        check_if_loaded)


class TextGenerationInput(TypedDict):
    system_prompt: Optional[str]
    user_prompt: str


class TextGenerationWithUniqueGroundTruth(TypedDict):
    x: TextGenerationInput
    y_true: str


class BigBenchHard(EvalCaseCollection):
    def __init__(
        self,
        name: str,
        dataset_name: str,
        split: str,
        subset: str,
        user_prompt_template: str,
    ) -> None:
        super().__init__(name)
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.user_prompt_template = user_prompt_template

    def _load(self) -> CollectionInfo:
        collection = datasets.load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
        )
        return CollectionInfo(
            collection=iter(collection),
            collection_len=collection.num_rows,
        )

    @check_if_loaded
    def __next__(self) -> TextGenerationWithUniqueGroundTruth:
        raw_item = next(self.collection)  # type: ignore
        return TextGenerationWithUniqueGroundTruth(
            x=TextGenerationInput(
                system_prompt=None,
                user_prompt=self.user_prompt_template.format(
                    original_input=raw_item["input"],
                ),
            ),
            y_true=raw_item["target"],
        )
