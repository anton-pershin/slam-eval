import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from rally.interaction import make_up_message_history

from slam_eval.performance.monitor import PerformanceMonitor
from slam_eval.performance.openai_collector import OpenAiStreamingCollector
from slam_eval.performance.storage import LocalPerformanceStorageAdapter
from slam_eval.utils.common import get_config_path

CONFIG_NAME = "config_main"
LOGGER = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    collection = instantiate(cfg.collection)
    scorer = instantiate(cfg.scorer)
    eval_storage_adapter = instantiate(cfg.storage_adapter)

    perf_cfg = cfg.get("performance_monitor")
    monitor: PerformanceMonitor | None = None
    perf_storage: LocalPerformanceStorageAdapter | None = None
    if perf_cfg is not None and perf_cfg.get("enabled", False):
        # D6: single source of truth — the YAML declares _target_; instantiate it.
        # `enabled` and `result_dir` are loop-level options, not constructor args.
        from omegaconf import open_dict

        monitor_kwargs = perf_cfg.copy()
        with open_dict(monitor_kwargs):
            monitor_kwargs.pop("enabled", None)
            monitor_kwargs.pop("result_dir", None)
        monitor = instantiate(monitor_kwargs)
        if hasattr(model, "llm"):  # OpenAI-compatible path
            # The collector measures through the model's own Llm, so the
            # request it sends carries the endpoint, the headers, the body and
            # the cap the model is configured with (FR5).
            monitor.openai_collector = OpenAiStreamingCollector(llm=model.llm)
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
        record_appended = False
        if monitor is not None and monitor.openai_collector is not None:
            # Single-request measurement (FR5/FR11/NFR1): the monitored
            # streaming request IS the prediction. No second request.
            # NOTE (N2): on this branch the collector OWNS the request —
            # model.predict() is never called, and the request is the Llm's
            # own (endpoint, generation parameters and cap included).
            result = monitor.openai_collector.measure(
                make_up_message_history(
                    system_prompt=eval_case["x"]["system_prompt"],
                    user_prompt=eval_case["x"]["user_prompt"],
                ),
                non_streaming=not monitor.streaming,
            )
            if result.get("streaming_failed"):
                monitor.note_streaming_fallback(
                    result["fallback_reason"],
                    warn=not monitor.streaming_fallback_intended,
                )
                monitor.note_openai_result(
                    case_index=i,
                    e2e_s=result.get("e2e_time_s"),  # None if no completed request
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
                    tpot_s=result.get("tpot_s"),
                )
                if result.get("tokens_source") == "chunk_count":
                    monitor.note_usage_fallback()
            y_pred = result.get("content") or ""
            record_appended = True
        else:
            y_pred = model.predict(eval_case["x"])
            if monitor is not None and hasattr(model, "step_callback"):
                # In-process path: the model renders its own prompt, so it is
                # the model that answers how many tokens that prompt takes (FR9).
                monitor.set_prompt_tokens(i, model.prompt_token_count(eval_case["x"]))
        if monitor is not None:
            monitor.on_prediction_end(i, record_already_appended=record_appended)
        if monitor is not None:
            monitor.on_scoring_start()
        score = scorer(eval_case["y_true"], y_pred)
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
        perf_storage.save_memory_samples(cfg.group_id, monitor.memory_samples)
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
