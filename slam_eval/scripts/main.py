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
        # D6: single source of truth — the YAML declares _target_; instantiate it.
        # `enabled` and `result_dir` are loop-level options, not constructor args.
        from omegaconf import open_dict

        monitor_kwargs = perf_cfg.copy()
        with open_dict(monitor_kwargs):
            monitor_kwargs.pop("enabled", None)
            monitor_kwargs.pop("result_dir", None)
        monitor = instantiate(monitor_kwargs)
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
        record_appended = False
        if monitor is not None and monitor.openai_collector is not None:
            # Single-request measurement (FR5/FR11/NFR1): the monitored
            # streaming request IS the prediction. No second request.
            # NOTE (N2): on this branch the collector OWNS the request —
            # model.predict() is never called. Any request-shaping logic
            # (system prompts, generation params) must be passed here
            # explicitly; changes inside the Model class will NOT apply.
            max_tokens = getattr(model.llm, "max_output_tokens", None)
            if max_tokens is None and hasattr(model.llm, "max_output_tokens"):
                raise ValueError(
                    "model.llm.max_output_tokens is set but resolved to None; "
                    "the collector would silently drop the generation cap"
                )
            result = monitor.openai_collector.measure(
                _build_messages(eval_case["x"]),
                max_output_tokens=max_tokens,
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
            if (
                monitor is not None
                and hasattr(model, "step_callback")
                and hasattr(model, "tokenizer")
            ):
                # In-process path: count prompt tokens for FR3
                messages = _build_messages(eval_case["x"])
                prompt_text = model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs=(
                        {"enable_thinking": model.enable_thinking}
                        if getattr(model, "enable_thinking", None) is not None
                        else {}
                    ),
                )
                prompt_ids = model.tokenizer(prompt_text)["input_ids"]
                monitor.set_prompt_tokens(i, len(prompt_ids))
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
