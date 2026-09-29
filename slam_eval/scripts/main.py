import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from slam_eval.performance.monitor import PerformanceMonitor
from slam_eval.performance.openai_collector import OpenAiStreamingCollector
from slam_eval.performance.storage import LocalPerformanceStorageAdapter
from slam_eval.utils.common import get_config_path

CONFIG_NAME = "config_main"
LOGGER = logging.getLogger(__name__)


def _build_messages(x) -> list[dict[str, str]]:
    messages = []
    if x["system_prompt"] is not None:
        messages.append({"role": "system", "content": x["system_prompt"]})
    messages.append({"role": "user", "content": x["user_prompt"]})
    return messages


def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    collection = instantiate(cfg.collection)
    scorer = instantiate(cfg.scorer)
    eval_storage_adapter = instantiate(cfg.storage_adapter)

    perf_cfg = cfg.get("performance_monitor")
    monitor: PerformanceMonitor | None = None
    perf_storage: LocalPerformanceStorageAdapter | None = None
    if perf_cfg is not None and perf_cfg.get("enabled", False):
        monitor = PerformanceMonitor(
            stats=list(
                perf_cfg.get("stats", ["min", "max", "mean", "median", "q5", "q95"])
            ),
            warmup_cases=perf_cfg.get("warmup_cases", 0),
            sampler_interval_s=perf_cfg.get("sampler_interval_s", 0.1),
            gpu_pids=perf_cfg.get("gpu_pids"),
            ram_pids=perf_cfg.get("ram_pids"),
            device_index=perf_cfg.get("device_index", 0),
            streaming=perf_cfg.get("streaming", True),
        )
        if hasattr(model, "llm"):  # OpenAI-compatible path
            llm = model.llm
            monitor.openai_collector = OpenAiStreamingCollector(
                url=llm.url,
                authorization=getattr(llm, "authorization", None),
                model=llm.model or "",
            )
        elif hasattr(model, "step_callback"):  # LocalCausalLm in-process path
            model.step_callback = monitor.make_step_callback()
        if perf_cfg.get("result_dir") is not None:
            perf_storage = LocalPerformanceStorageAdapter(
                result_dir=perf_cfg.result_dir
            )
        LOGGER.info("Performance monitoring enabled")
        monitor.on_run_start(cfg.group_id)

    collection.load()
    model_answers = []
    scores = []

    collection_length = len(collection)
    for i, eval_case in enumerate(collection):
        LOGGER.info("Run test case #%s out of %s", i + 1, collection_length)
        if monitor is not None:
            monitor.on_prediction_start(i)
        y_pred = model.predict(eval_case["x"])
        if monitor is not None:
            if monitor.openai_collector is not None:
                messages = _build_messages(eval_case["x"])
                result = monitor.openai_collector.measure(
                    messages,
                    max_output_tokens=getattr(model.llm, "max_output_tokens", None),
                )
                if result.get("streaming_failed"):
                    monitor.note_streaming_fallback(
                        result["fallback_reason"],
                        warn=not monitor.streaming_fallback_intended,
                    )
                    monitor.note_openai_result(
                        case_index=i,
                        e2e_s=result.get("e2e_time_s") or 0.0,
                        ttft_s=None,
                        generated_tokens=None,
                        prompt_tokens=None,
                    )
                else:
                    monitor.note_openai_result(
                        case_index=i,
                        e2e_s=result["e2e_time_s"],
                        ttft_s=result["ttft_s"],
                        generated_tokens=result["generated_tokens"],
                        prompt_tokens=result["prompt_tokens"],
                    )
                if result.get("tokens_source") == "chunk_count":
                    monitor.note_usage_fallback()
                y_pred = result.get("content") or y_pred
            monitor.on_prediction_end(i)
        score = scorer(eval_case["y_true"], y_pred)
        if monitor is not None:
            monitor.on_scoring_start()
        model_answers.append(y_pred)
        scores.append(score)
        if monitor is not None:
            monitor.on_scoring_end()

    eval_storage_adapter.save(
        group_id=cfg.group_id,
        model=model,
        eval_case_collection=collection,
        scores=scores,
        model_answers=model_answers,
    )

    if monitor is not None and perf_storage is not None:
        aggregated = monitor.on_run_end(
            cfg.group_id,
            run_metadata={"group_id": cfg.group_id, "model": model.name},
        )
        perf_storage.save_raw(cfg.group_id, monitor.raw_records)
        perf_storage.save_aggregated(cfg.group_id, aggregated)
        LOGGER.info(
            "Performance artifacts saved under %s",
            perf_storage.run_dir(cfg.group_id),
        )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(main)()
