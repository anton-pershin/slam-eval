import json
from typing import Any
from abc import ABC, abstractmethod
from datetime import datetime

from slam_eval.model import Model
from slam_eval.collection import EvalCaseCollection
from slam_eval.utils.typing import HasStr


class EvalStorageAdapter(ABC):
    def __init__(self) -> None:
        pass

    def save(
        self,
        model: Model,
        eval_case_collection: EvalCaseCollection,
        scores: list[int | float],
        model_answers: list[HasStr],
        **other_results
    ) -> None:
        datetime_now = datetime.now()
        result_dict = {
            "eval_id": "{datetime}_M_{model}_C_{eval_case_collection}".format(
                datetime=datetime_now.isoformat("_"),
                model=model.name,
                eval_case_collection=eval_case_collection.name
            ),
            "timestamp": datetime_now.timestamp(),
            "model": model.name,
            "eval_case_collection": eval_case_collection.name,
            "scores": scores,
            "model_answers": model_answers
        }

        for k, v in other_results.items():
            result_dict[k] = v

        self._save_result_dict(result_dict)

    @abstractmethod
    def _save_result_dict(self, result_dict: dict[str, str]) -> None:
        ...


class LocalJsonlAdapter(EvalStorageAdapter):
    def __init__(self, path_to_jsonl: str) -> None:
        super().__init__()
        self.path_to_jsonl = path_to_jsonl

    def _save_result_dict(self, result_dict: dict[str, str]) -> None:
        with open(self.path_to_jsonl, "a") as f:
            f.write(json.dumps(result_dict) + "\n")
