"""Tests for slam-eval performance monitoring (spec 04, AC1-AC9)."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any

import pytest
from rally.interaction import make_up_message_history
from rally.llm import (
    LlmAuthorizationError,
    LlmStreamEvent,
    LlmStreamRejectedError,
    LlmTimeoutError,
    LlmTransportError,
    LlmUsage,
)
from slam_core.collections.text_generation import TextGenerationInput
from slam_core.model import LlmViaOpenAiApi

from slam_eval.performance.monitor import PHASE_PREDICT, PHASE_SCORE, PerformanceMonitor
from slam_eval.performance.openai_collector import OpenAiStreamingCollector
from slam_eval.performance.sampler import MemorySampler
from slam_eval.performance.stats import aggregate, resolve_stat
from slam_eval.performance.storage import LocalPerformanceStorageAdapter

sys.path.insert(0, "/home/tony/reps/github/anton-pershin/slam-core")


class StubLlm:
    """A `Llm` double: scripted events with arrival delays, calls recorded."""

    url = "http://stub.example/v1/chat/completions"
    model = "stub-model"

    def __init__(self, events=(), error=None, message=None):
        self.events = list(events)  # (delay_before_event, LlmStreamEvent)
        self.error = error
        self.message = message
        self.stream_calls: list[list[dict[str, str]]] = []
        self.request_calls: list[list[dict[str, str]]] = []

    def stream(self, messages):
        self.stream_calls.append(messages)
        if self.error is not None:
            raise self.error
        for delay, event in self.events:
            time.sleep(delay)
            yield event

    def request(self, messages):
        self.request_calls.append(messages)
        return self.message


MESSAGES = [{"role": "user", "content": "hi"}]


class TestStatsRegistry:
    def test_default_stats(self):
        values = [1.0, 2.0, 3.0, 4.0, 10.0]
        agg = aggregate(values, ["min", "max", "mean", "median", "q5", "q95"])
        assert agg["n"] == 5
        assert agg["min"] == 1.0
        assert agg["max"] == 10.0
        assert agg["mean"] == 4.0
        assert agg["median"] == 3.0

    def test_nulls_excluded(self):
        agg = aggregate([1.0, None, 3.0], ["min", "max", "mean"])
        assert agg["n"] == 2
        assert agg["mean"] == 2.0

    def test_all_null_yields_zero_n_with_none_entries(self):
        agg = aggregate([None, None], ["min", "mean"])
        assert agg["n"] == 0
        assert agg["min"] is None
        assert agg["mean"] is None

    def test_configurable_non_default_stat_q99(self):
        agg = aggregate([float(i) for i in range(100)], ["q99"])
        assert agg["n"] == 100
        assert agg["q99"] == pytest.approx(98.01, abs=1e-6)

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            resolve_stat("variance")


class TestMonitorLifecycle:
    def test_full_lifecycle_records_per_case(self):
        monitor = PerformanceMonitor(warmup_cases=0)
        monitor.on_run_start("g")
        for i in range(3):
            monitor.on_prediction_start(i)
            time.sleep(0.01)
            monitor.on_prediction_end(i)
            monitor.on_scoring_start()
            monitor.on_scoring_end()
        agg = monitor.on_run_end("g", run_metadata={"model": "m"})
        raw = monitor.raw_records
        assert len(raw) == 3
        assert [r["case_id"] for r in raw] == [0, 1, 2]
        assert all(r["e2e_time_s"] > 0 for r in raw)
        assert all(r["warmup"] is False for r in raw)
        assert agg["e2e_time_s"]["n"] == 3

    def test_warmup_excluded_from_aggregates_kept_in_raw(self):
        monitor = PerformanceMonitor(warmup_cases=1)
        for i in range(3):
            monitor.on_prediction_start(i)
            monitor.on_prediction_end(i)
        monitor.on_run_end("g", run_metadata={})
        raw = monitor.raw_records
        assert [r["warmup"] for r in raw] == [True, False, False]
        agg = (
            monitor.on_run_end("g", run_metadata={})
            if False
            else monitor.build_aggregated("g", {})
        )
        assert agg["e2e_time_s"]["n"] == 2

    def test_tpot_null_semantics(self):
        monitor = PerformanceMonitor()
        # <2 tokens => tpot null
        monitor.on_prediction_start(0)
        monitor._token_state.on_step()  # 1 token
        monitor.on_prediction_end(0)
        assert monitor.raw_records[0]["tpot_s"] is None
        assert monitor.raw_records[0]["ttft_s"] is not None

    def test_streaming_fallback_signals(self, caplog):
        monitor = PerformanceMonitor()
        with caplog.at_level(logging.WARNING):
            monitor.note_streaming_fallback("streaming_request_rejected", warn=True)
        assert any("Streaming unavailable" in r.message for r in caplog.records)
        monitor.note_streaming_fallback("streaming_request_rejected", warn=True)
        agg = monitor.build_aggregated("g", {})
        assert agg["run_metadata"]["streaming"]["supported"] is False
        assert (
            agg["run_metadata"]["streaming"]["fallback_reason"]
            == "streaming_request_rejected"
        )
        # warnings only once
        warn_count = sum(
            1 for r in caplog.records if "Streaming unavailable" in r.message
        )
        assert warn_count == 1

    def test_disabled_in_config_no_warning(self, caplog):
        monitor = PerformanceMonitor(streaming=False)
        with caplog.at_level(logging.WARNING):
            monitor.note_streaming_fallback("disabled_in_config", warn=False)
        assert not [r for r in caplog.records if "Streaming unavailable" in r.message]


class TestMemorySampler:
    def test_samples_phase_tagged(self):
        phase_holder = ["idle"]
        sampler = MemorySampler(
            interval_s=0.01,
            phase_getter=lambda: phase_holder[0],
            ram_pids=[os.getpid()],
        )
        sampler.start()
        phase_holder[0] = PHASE_PREDICT
        time.sleep(0.05)
        phase_holder[0] = PHASE_SCORE
        time.sleep(0.05)
        sampler.stop()
        samples = sampler.drain_samples()
        phases = {s["phase"] for s in samples}
        assert PHASE_PREDICT in phases
        assert PHASE_SCORE in phases
        assert all(s["rss_bytes"] > 0 for s in samples)

    def test_no_targets_no_start(self):
        sampler = MemorySampler(interval_s=0.01, phase_getter=lambda: "idle")
        sampler.start()
        time.sleep(0.03)
        sampler.stop()
        assert sampler.drain_samples() == []

    def test_dead_pid_degrades_gracefully(self):
        sampler = MemorySampler(
            interval_s=0.01, phase_getter=lambda: "predict", ram_pids=[-1]
        )
        sampler.start()
        time.sleep(0.03)
        sampler.stop()
        samples = sampler.drain_samples()
        assert samples  # thread alive
        # NFR5: unavailable PID is null, never a misleading integer 0
        assert all(s["rss_bytes"] is None for s in samples)

    def test_pynvml_blocked_never_crashes(self, monkeypatch):
        import slam_eval.performance.sampler as sampler_mod

        monkeypatch.setattr(sampler_mod, "_HAS_PYNVML", False)
        sampler = MemorySampler(
            interval_s=0.01,
            phase_getter=lambda: "predict",
            gpu_pids=[os.getpid()],
            ram_pids=[os.getpid()],
        )
        sampler.start()
        time.sleep(0.03)
        sampler.stop()
        samples = sampler.drain_samples()
        assert all(s["vram_bytes"] is None for s in samples)
        assert all(s["rss_bytes"] > 0 for s in samples)


class TestCollectorThroughTheLlm:
    """FR5/FR7: the collector measures through the Llm it is given and owns
    nothing else — no url, no headers, no body, no framing, no dialect."""

    def test_collector_calls_the_llms_stream_with_the_messages(self):
        llm = StubLlm(events=[(0.0, LlmStreamEvent(content="hi"))])
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert llm.stream_calls == [MESSAGES]
        assert llm.request_calls == []
        assert result["streaming_failed"] is False

    def test_non_streaming_measurement_requests_through_the_llm(self):
        llm = StubLlm(message={"role": "assistant", "content": "Hi"})
        result = OpenAiStreamingCollector(llm).measure(MESSAGES, non_streaming=True)
        assert llm.request_calls == [MESSAGES]
        assert llm.stream_calls == []
        assert result["streaming_failed"] is False
        assert result["content"] == "Hi"
        assert result["ttft_s"] is None and result["tpot_s"] is None
        # rally's request() answers with the message only: no usage to report
        assert result["prompt_tokens"] is None
        assert result["generated_tokens"] is None

    def test_ttft_and_tpot_come_from_event_arrival(self):
        llm = StubLlm(
            events=[
                (0.05, LlmStreamEvent(content="Hello")),
                (0.02, LlmStreamEvent(content=" world")),
                (0.02, LlmStreamEvent(content="!")),
            ]
        )
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert 0.03 <= result["ttft_s"] <= 0.2
        assert result["content"] == "Hello world!"
        assert 0.0 < result["tpot_s"] < result["e2e_time_s"]

    def test_usage_is_preferred_over_content_event_counting(self):
        llm = StubLlm(
            events=[
                (0.01, LlmStreamEvent(content="Hello")),
                (0.01, LlmStreamEvent(content=" world")),
                (
                    0.01,
                    LlmStreamEvent(
                        content="!",
                        usage=LlmUsage(prompt_tokens=10, completion_tokens=7),
                    ),
                ),
            ]
        )
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["tokens_source"] == "usage"
        assert result["prompt_tokens"] == 10
        assert result["generated_tokens"] == 7

    def test_absent_usage_falls_back_to_content_event_counting(self):
        llm = StubLlm(
            events=[
                (0.01, LlmStreamEvent(content="a")),
                (0.01, LlmStreamEvent(content="b")),
            ]
        )
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["tokens_source"] == "chunk_count"
        assert result["generated_tokens"] == 2
        assert result["prompt_tokens"] is None

    def test_authorization_error_aborts(self):
        llm = StubLlm(error=LlmAuthorizationError("HTTP 401"))
        with pytest.raises(RuntimeError, match="Authorization failed"):
            OpenAiStreamingCollector(llm).measure(MESSAGES)

    def test_rejected_streaming_request_falls_back(self):
        llm = StubLlm(error=LlmStreamRejectedError(400, "no stream"))
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["streaming_failed"] is True
        assert result["fallback_reason"] == "streaming_request_rejected"

    def test_transport_error_falls_back_without_e2e(self):
        llm = StubLlm(error=LlmTransportError("conn reset"))
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["streaming_failed"] is True
        assert result["e2e_time_s"] is None

    def test_timeout_gives_the_same_record_as_a_transport_error(self):
        llm = StubLlm(error=LlmTimeoutError("no data for 30s"))
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["streaming_failed"] is True
        assert result["fallback_reason"] == "streaming_request_rejected"
        assert result["e2e_time_s"] is None

    def test_truncated_stream_is_the_completed_answer(self):
        """FR8: the cap cut the answer short — what arrived IS the answer."""
        llm = StubLlm(
            events=[
                (0.01, LlmStreamEvent(reasoning="thinking out loud")),
                (0.0, LlmStreamEvent(finish_reason="length", truncated=True)),
            ]
        )
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["streaming_failed"] is False
        assert result["content"] == ""
        assert result["ttft_s"] is None  # no content event ever arrived

    def test_truncated_stream_with_partial_content_keeps_the_content(self):
        llm = StubLlm(
            events=[
                (0.01, LlmStreamEvent(content='{"name"')),
                (0.0, LlmStreamEvent(finish_reason="length", truncated=True)),
            ]
        )
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["streaming_failed"] is False
        assert result["content"] == '{"name"'
        assert result["ttft_s"] is not None

    def test_no_content_and_not_truncated_reports_unavailability(self):
        llm = StubLlm(events=[(0.01, LlmStreamEvent(reasoning="thinking only"))])
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["streaming_failed"] is True
        assert result["fallback_reason"] == "streaming_request_rejected"

    def test_non_streaming_failure_record_shape(self):
        """NFR4: rally's request() answers None for every failure; the record
        says a fallback happened, keeps no timing and names no reason."""
        llm = StubLlm(message=None)
        result = OpenAiStreamingCollector(llm).measure(MESSAGES, non_streaming=True)
        assert result["streaming_failed"] is True
        assert result["e2e_time_s"] is None
        # rally's non-streaming request cannot tell the failures apart, so the
        # record names no reason at all (row 18)
        assert result.get("fallback_reason") is None

    def test_thinking_trace_in_content_reaches_y_pred_verbatim(self):
        llm = StubLlm(
            events=[(0.0, LlmStreamEvent(content="<think>hmm</think>answer"))]
        )
        result = OpenAiStreamingCollector(llm).measure(MESSAGES)
        assert result["content"] == "<think>hmm</think>answer"

    def test_collector_and_predict_send_the_same_messages(self):
        llm = StubLlm(
            events=[(0.0, LlmStreamEvent(content="ok"))],
            message={"role": "assistant", "content": "ok"},
        )
        model = LlmViaOpenAiApi("m", llm)
        x = TextGenerationInput(system_prompt="be nice", user_prompt="hello")

        OpenAiStreamingCollector(llm).measure(
            make_up_message_history(
                system_prompt=x["system_prompt"], user_prompt=x["user_prompt"]
            )
        )
        model.predict(x)

        assert llm.stream_calls == llm.request_calls
        assert llm.stream_calls == [
            [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hello"},
            ]
        ]


class TestStorage:
    def test_local_storage_layout(self, tmp_path):
        storage = LocalPerformanceStorageAdapter(result_dir=str(tmp_path))
        raw_path = storage.save_raw(
            "g1",
            [
                {
                    "case_id": 0,
                    "e2e_time_s": 1.0,
                    "ttft_s": None,
                    "tpot_s": None,
                    "prompt_tokens": None,
                    "generated_tokens": None,
                    "warmup": False,
                }
            ],
        )
        agg_path = storage.save_aggregated(
            "g1",
            {"e2e_time_s": {"n": 1, "min": 1.0}, "run_metadata": {"model": "m"}},
        )
        assert os.path.basename(raw_path) == "raw.jsonl"
        assert os.path.basename(agg_path) == "aggregated.json"
        lines = open(raw_path).read().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["case_id"] == 0
        agg = json.load(open(agg_path))
        assert agg["e2e_time_s"]["n"] == 1
        assert "performance_g1" in raw_path


class TestRunKeyTimestamp:
    def test_timestamped_run_key_no_collision(self, tmp_path):
        """FR10: two runs with the same group_id must not overwrite each other."""
        s1 = LocalPerformanceStorageAdapter(result_dir=str(tmp_path))
        s2 = LocalPerformanceStorageAdapter(result_dir=str(tmp_path))
        p1 = s1.save_raw("g1", [{"case_id": 0}])
        p2 = s2.save_raw("g1", [{"case_id": 0}, {"case_id": 1}])
        assert p1 != p2
        assert len(open(p1).read().strip().splitlines()) == 1
        assert len(open(p2).read().strip().splitlines()) == 2


class TestSingleRequestOpenAiPath:
    """FR11/NFR1: the measured request IS the prediction; one record per case."""

    def test_record_already_appended_no_double(self):
        monitor = PerformanceMonitor()
        monitor.on_prediction_start(0)
        monitor.note_openai_result(
            case_index=0,
            e2e_s=1.0,
            ttft_s=0.1,
            generated_tokens=10,
            prompt_tokens=5,
            tpot_s=0.09,
        )
        monitor.on_prediction_end(0, record_already_appended=True)
        assert len(monitor.raw_records) == 1  # no duplicate

    def test_collector_tpot_preferred(self):
        monitor = PerformanceMonitor()
        monitor.on_prediction_start(0)
        monitor.note_openai_result(
            case_index=0,
            e2e_s=2.0,
            ttft_s=0.5,
            generated_tokens=10,
            prompt_tokens=5,
            tpot_s=0.111,  # chunk-delta value is authoritative
        )
        monitor.on_prediction_end(0, record_already_appended=True)
        assert monitor.raw_records[0]["tpot_s"] == 0.111

    def test_phase_tag_order_scoring(self):
        """FR7: scorer runs INSIDE the score phase (loop order fixed)."""
        monitor = PerformanceMonitor()
        monitor.on_prediction_start(0)
        monitor.on_prediction_end(0)
        monitor.on_scoring_start()
        assert monitor._phase == "score"  # scorer executes here in main.py
        monitor.on_scoring_end()
        assert monitor._phase == "idle"


class TestStreamingDisabledMetadata:
    def test_disabled_in_config_reflected_at_construction(self):
        """FR5 signal (a): streaming=false -> supported=False, reason set immediately."""
        from slam_eval.performance.monitor import PerformanceMonitor

        monitor = PerformanceMonitor(streaming=False)
        agg = monitor.build_aggregated("g", {})
        state = agg["run_metadata"]["streaming"]
        assert state["supported"] is False
        assert state["fallback_reason"] == "disabled_in_config"

    def test_streaming_true_default(self):
        from slam_eval.performance.monitor import PerformanceMonitor

        monitor = PerformanceMonitor(streaming=True)
        agg = monitor.build_aggregated("g", {})
        state = agg["run_metadata"]["streaming"]
        assert state["supported"] is True
        assert state["fallback_reason"] is None
