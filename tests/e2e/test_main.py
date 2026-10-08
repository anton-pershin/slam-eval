import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import hydra
import pytest
from freezegun import freeze_time
from omegaconf import DictConfig, OmegaConf
from rally.llm import Llm, LlmStreamEvent
from slam_core.collections.base import CollectionInfo, EvalCase, EvalCaseCollection
from slam_core.collections.text_generation import TextGenerationInput
from slam_core.model import Model
from slam_core.scorers.base import Score, Scorer
from slam_core.storage_adapter import EvalStorageAdapter
from slam_core.utils.common import get_config_path
from slam_core.utils.typing import HasStr

from slam_eval.scripts.main import main

DICT_STORAGE = []


class SimpleEvalCaseCollection(EvalCaseCollection):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.i = 0

    def _load(self) -> CollectionInfo:
        self.collection_data = [
            ("Test question 1", "Test answer 1"),
            ("Test question 2", "Test answer 2"),
            ("Test question 3", "Test answer 3"),
        ]
        return CollectionInfo(
            collection=iter(self.collection_data),
            collection_len=len(self.collection_data),
        )

    def __next__(self) -> EvalCase:
        if self.i >= len(self.collection_data):
            raise StopIteration

        res = self.collection_data[self.i]
        self.i += 1
        return {
            "x": TextGenerationInput(system_prompt=None, user_prompt=res[0]),
            "y_true": res[1],
        }


class SimpleScoreScorer(Scorer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, y_true, y_pred) -> Score:
        return Score(primary=float(y_true == y_pred), sub_scores=None)


class ComplexScoreScorer(Scorer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, y_true, y_pred) -> Score:
        exact = float(y_true == y_pred)
        non_empty = float(bool(y_pred))
        primary = (exact + non_empty) / 2
        return Score(
            primary=primary,
            sub_scores={
                "exact_match": exact,
                "non_empty": non_empty,
            },
        )


class SimpleEvalStorageAdapter(EvalStorageAdapter):
    def __init__(self) -> None:
        global DICT_STORAGE
        self.dict_storage: list[dict[str, Any]] = DICT_STORAGE

    def load(self, id_regex: str) -> list[dict[str, Any]]:
        """Load evaluation results filtered by regex pattern on id field."""
        import re

        pattern = re.compile(id_regex)
        results = []

        for result_dict in self.dict_storage:
            if "id" in result_dict and pattern.search(result_dict["id"]):
                results.append(result_dict)

        return results

    def _save_result_dict(self, result_id: str, result_dict: dict[str, Any]) -> None:
        result_dict_with_id = {"id": result_id, **result_dict}
        self.dict_storage.append(result_dict_with_id)


@pytest.fixture
def cfg():
    with hydra.initialize(
        version_base="1.3", config_path="../../config", job_name="test_app"
    ):
        default_cfg = hydra.compose(config_name="config_main")

    return default_cfg


@pytest.fixture
def eval_case_collection_cfg():
    return {
        "_target_": "tests.e2e.test_main.SimpleEvalCaseCollection",
        "name": "simple_eval_case_collection",
    }


@pytest.fixture
def storage_adapter_cfg():
    return {
        "_target_": "tests.e2e.test_main.SimpleEvalStorageAdapter",
    }


@pytest.fixture
def simple_scorer_cfg():
    return {
        "_target_": "tests.e2e.test_main.SimpleScoreScorer",
        "name": "simple_score_scorer",
    }


@pytest.fixture
def complex_scorer_cfg():
    return {
        "_target_": "tests.e2e.test_main.ComplexScoreScorer",
        "name": "complex_score_scorer",
    }


@pytest.fixture(autouse=True)
def reset_dict_storage():
    """Reset the global DICT_STORAGE before each test."""
    global DICT_STORAGE
    DICT_STORAGE.clear()
    yield
    DICT_STORAGE.clear()


@freeze_time("2000-01-01")
def test_main_with_simple_scorer(
    cfg: DictConfig,
    eval_case_collection_cfg,
    storage_adapter_cfg,
    simple_scorer_cfg,
    monkeypatch,
):
    monkeypatch.setattr(
        "rally.llm.Llm.request",
        lambda *args, **kwargs: {"role": "assistant", "content": "Test answer 1"},
    )

    cfg.collection = eval_case_collection_cfg
    cfg.storage_adapter = storage_adapter_cfg
    cfg.scorer = simple_scorer_cfg

    main(cfg)

    datetime_now = datetime.datetime.now()

    global DICT_STORAGE
    assert DICT_STORAGE == [
        {
            "id": "eval:{group_id}:{datetime}_M_{model}_C_{eval_case_collection}".format(
                group_id=cfg.group_id,
                datetime=datetime_now.isoformat("_"),
                model=cfg.model.name,
                eval_case_collection=cfg.collection.name,
            ),
            "group_id": cfg.group_id,
            "timestamp": datetime_now.timestamp(),
            "model": cfg.model.name,
            "eval_case_collection": cfg.collection.name,
            "scores": [1.0, 0.0, 0.0],
            "sub_scores": [None, None, None],
            "model_answers": ["Test answer 1"] * 3,
        }
    ]


@freeze_time("2000-01-01")
def test_main_with_complex_scorer(
    cfg: DictConfig,
    eval_case_collection_cfg,
    storage_adapter_cfg,
    complex_scorer_cfg,
    monkeypatch,
):
    monkeypatch.setattr(
        "rally.llm.Llm.request",
        lambda *args, **kwargs: {"role": "assistant", "content": "Test answer 1"},
    )

    cfg.collection = eval_case_collection_cfg
    cfg.storage_adapter = storage_adapter_cfg
    cfg.scorer = complex_scorer_cfg

    main(cfg)

    datetime_now = datetime.datetime.now()

    global DICT_STORAGE
    assert DICT_STORAGE == [
        {
            "id": "eval:{group_id}:{datetime}_M_{model}_C_{eval_case_collection}".format(
                group_id=cfg.group_id,
                datetime=datetime_now.isoformat("_"),
                model=cfg.model.name,
                eval_case_collection=cfg.collection.name,
            ),
            "group_id": cfg.group_id,
            "timestamp": datetime_now.timestamp(),
            "model": cfg.model.name,
            "eval_case_collection": cfg.collection.name,
            "scores": [1.0, 0.5, 0.5],
            "sub_scores": [
                {"exact_match": 1.0, "non_empty": 1.0},
                {"exact_match": 0.0, "non_empty": 1.0},
                {"exact_match": 0.0, "non_empty": 1.0},
            ],
            "model_answers": ["Test answer 1"] * 3,
        }
    ]


def _enabled_perf_cfg(tmp_path):
    """The enabled monitoring config, writing its artifacts under tmp_path."""
    return OmegaConf.create(
        {
            "_target_": "slam_eval.performance.monitor.PerformanceMonitor",
            "enabled": True,
            "stats": ["min", "max", "mean", "median", "q5", "q95"],
            "warmup_cases": 0,
            "sampler_interval_s": 0.1,
            "self_monitor": False,
            "gpu_pids": None,
            "ram_pids": None,
            "device_index": 0,
            "streaming": True,
            "result_dir": str(tmp_path),
        }
    )


def _raw_records(result_dir):
    (path,) = Path(result_dir).rglob("raw.jsonl")
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


@freeze_time("2000-01-01")
def test_monitoring_enabled_run_completes_without_a_cap(
    cfg: DictConfig,
    eval_case_collection_cfg,
    storage_adapter_cfg,
    simple_scorer_cfg,
    tmp_path,
    monkeypatch,
):
    """FR8/rows 4-5: no cap to guard, one measured record per case."""
    monkeypatch.setattr(
        "rally.llm.Llm.stream",
        lambda self, messages: iter([LlmStreamEvent(content="Test answer 1")]),
    )
    cfg.collection = eval_case_collection_cfg
    cfg.storage_adapter = storage_adapter_cfg
    cfg.scorer = simple_scorer_cfg
    cfg.performance_monitor = _enabled_perf_cfg(tmp_path)
    assert cfg.model.llm.get("max_output_tokens") is None  # row 4's premise

    main(cfg)

    records = _raw_records(tmp_path)
    assert [r["case_id"] for r in records] == [0, 1, 2]
    assert all(r["e2e_time_s"] is not None for r in records)
    # no usage was reported, so the counts come from counting content events
    assert all(r["generated_tokens"] == 1 for r in records)
    assert DICT_STORAGE[0]["model_answers"] == ["Test answer 1"] * 3

    (aggregated_path,) = Path(tmp_path).rglob("aggregated.json")
    metadata = json.loads(aggregated_path.read_text())["run_metadata"]
    assert (
        metadata["streaming"]["fallback_reason"]
        == "usage_not_reported_chunk_counting_used"
    )
    assert metadata["streaming"]["supported"] is True


@freeze_time("2000-01-01")
def test_collector_is_built_from_the_models_llm(
    cfg: DictConfig,
    eval_case_collection_cfg,
    storage_adapter_cfg,
    simple_scorer_cfg,
    tmp_path,
    monkeypatch,
):
    """FR5: `OpenAiStreamingCollector(llm)` — no url/authorization/model args."""
    import slam_eval.scripts.main as main_module

    streamed_by = []

    def record_stream(self, messages):
        streamed_by.append(self)
        return iter([LlmStreamEvent(content="Test answer 1")])

    monkeypatch.setattr("rally.llm.Llm.stream", record_stream)
    created = []
    real_collector = main_module.OpenAiStreamingCollector

    class Recording(real_collector):
        def __init__(self, llm):
            super().__init__(llm)
            created.append(llm)

    monkeypatch.setattr(main_module, "OpenAiStreamingCollector", Recording)
    cfg.collection = eval_case_collection_cfg
    cfg.storage_adapter = storage_adapter_cfg
    cfg.scorer = simple_scorer_cfg
    cfg.performance_monitor = _enabled_perf_cfg(tmp_path)

    main(cfg)

    assert len(created) == 1
    assert isinstance(created[0], Llm)
    assert created[0].url == cfg.model.llm.url
    assert created[0].model == cfg.model.llm.model
    assert streamed_by and all(llm is created[0] for llm in streamed_by)


class FakeInProcessModel(Model):
    """An in-process model that knows its own prompt token count."""

    def __init__(self, name: str = "fake_local", prompt_tokens: int = 7) -> None:
        super().__init__(name)
        self.prompt_tokens = prompt_tokens
        self.step_callback = None

        def refuse(*args, **kwargs):
            raise AssertionError("the loop must not render the prompt itself (FR12)")

        self.tokenizer = SimpleNamespace(apply_chat_template=refuse)

    def prompt_token_count(self, x) -> int:
        return self.prompt_tokens

    def predict(self, x) -> str:
        if self.step_callback is not None:
            self.step_callback(0)
        return "Test answer 1"


@freeze_time("2000-01-01")
def test_in_process_run_counts_prompt_tokens_from_the_model(
    cfg: DictConfig,
    eval_case_collection_cfg,
    storage_adapter_cfg,
    simple_scorer_cfg,
    tmp_path,
):
    """FR9/FR12: the model answers the count; the loop renders nothing."""
    cfg.model = {
        "_target_": "tests.e2e.test_main.FakeInProcessModel",
        "name": "fake_local",
    }
    cfg.collection = eval_case_collection_cfg
    cfg.storage_adapter = storage_adapter_cfg
    cfg.scorer = simple_scorer_cfg
    cfg.performance_monitor = _enabled_perf_cfg(tmp_path)

    main(cfg)

    records = _raw_records(tmp_path)
    assert records[0]["prompt_tokens"] == 7
    # the clock is frozen, so the measured wall time is exactly zero — the
    # point is that an in-process record carries one at all
    assert records[0]["e2e_time_s"] is not None
