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

from slam_eval.performance.monitor import PHASE_PREDICT, PHASE_SCORE, PerformanceMonitor
from slam_eval.performance.openai_collector import OpenAiStreamingCollector
from slam_eval.performance.sampler import MemorySampler
from slam_eval.performance.stats import aggregate, resolve_stat
from slam_eval.performance.storage import LocalPerformanceStorageAdapter

sys.path.insert(0, "/home/tony/reps/github/anton-pershin/slam-core")


class FakeStreamResponse:
    """Simulates a streaming SSE response with per-chunk delays."""

    def __init__(self, chunks: list[tuple[float, str]], usage: dict | None = None):
        self._chunks = chunks  # (delay_before_chunk, content)
        self._usage = usage

    def __iter__(self):
        return self._gen()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def _gen(self):
        import json as _json

        for delay, content in self._chunks:
            time.sleep(delay)
            chunk: dict[str, Any] = {
                "choices": [{"delta": {"content": content}}],
            }
            if self._usage is not None and content == self._chunks[-1][1]:
                chunk["usage"] = self._usage
            yield ("data: " + _json.dumps(chunk) + "\n\n").encode()
        yield b"data: [DONE]\n\n"


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


class TestOpenAiStreamingCollector:
    def _patch_urlopen(self, monkeypatch, response):
        import slam_eval.performance.openai_collector as mod

        monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req: response)

    def test_streaming_ttft_and_usage_preferred(self, monkeypatch):
        chunks = [(0.05, "Hello"), (0.02, " world"), (0.02, "!")]
        usage = {"prompt_tokens": 10, "completion_tokens": 7}
        self._patch_urlopen(monkeypatch, FakeStreamResponse(chunks, usage))
        collector = OpenAiStreamingCollector("http://fake", None, "m")
        result = collector.measure([{"role": "user", "content": "hi"}])
        assert result["streaming_failed"] is False
        assert 0.04 <= result["ttft_s"] <= 0.2  # ~0.05 s first-chunk delay
        assert result["prompt_tokens"] == 10
        assert result["generated_tokens"] == 7  # usage preferred over 3 chunks
        assert result["tokens_source"] == "usage"
        assert result["content"] == "Hello world!"
        assert result["tpot_s"] is not None
        # magnitude sanity: TPOT is (last chunk elapsed - ttft) / (n-1), bounded
        # by e2e; a clock-mixup bug would produce absurd values (regression guard)
        assert 0.0 < result["tpot_s"] < result["e2e_time_s"]

    def test_chunk_count_fallback(self, monkeypatch):
        chunks = [(0.01, "a"), (0.01, "b")]
        self._patch_urlopen(monkeypatch, FakeStreamResponse(chunks, usage=None))
        collector = OpenAiStreamingCollector("http://fake", None, "m")
        result = collector.measure([{"role": "user", "content": "hi"}])
        assert result["tokens_source"] == "chunk_count"
        assert result["generated_tokens"] == 2  # approximate
        assert result["prompt_tokens"] is None

    def test_streaming_rejected(self, monkeypatch):
        import urllib.error

        import slam_eval.performance.openai_collector as mod

        def raise_http(req):
            raise urllib.error.HTTPError(req.full_url, 400, "no stream", None, None)

        monkeypatch.setattr(mod.urllib.request, "urlopen", raise_http)
        collector = OpenAiStreamingCollector("http://fake", None, "m")
        result = collector.measure([{"role": "user", "content": "hi"}])
        assert result["streaming_failed"] is True
        assert result["fallback_reason"] == "streaming_request_rejected"


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

    def test_non_streaming_branch(self, monkeypatch):
        import slam_eval.performance.openai_collector as mod

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": "Hi"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    }
                ).encode()

        monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req: FakeResponse())
        collector = OpenAiStreamingCollector("http://fake", None, "m")
        result = collector.measure(
            [{"role": "user", "content": "hi"}], non_streaming=True
        )
        assert result["streaming_failed"] is False
        assert result["content"] == "Hi"
        assert result["ttft_s"] is None and result["tpot_s"] is None
        assert result["prompt_tokens"] == 3 and result["generated_tokens"] == 2

    def test_phase_tag_order_scoring(self):
        """FR7: scorer runs INSIDE the score phase (loop order fixed)."""
        monitor = PerformanceMonitor()
        monitor.on_prediction_start(0)
        monitor.on_prediction_end(0)
        monitor.on_scoring_start()
        assert monitor._phase == "score"  # scorer executes here in main.py
        monitor.on_scoring_end()
        assert monitor._phase == "idle"


class TestAuthFailure:
    def test_auth_failure_raises(self, monkeypatch):
        """401/403 => RuntimeError, never a silent metrics fallback."""
        import urllib.error

        import slam_eval.performance.openai_collector as mod

        def raise_401(req):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", None, None)

        monkeypatch.setattr(mod.urllib.request, "urlopen", raise_401)
        collector = OpenAiStreamingCollector("http://fake", "Bearer bad", "m")
        with pytest.raises(RuntimeError, match="Authorization failed"):
            collector.measure([{"role": "user", "content": "hi"}])

    def test_failed_request_no_fabricated_e2e(self, monkeypatch):
        """A failed request records no e2e — null, never 0.0 (NFR5)."""
        import urllib.error

        import slam_eval.performance.openai_collector as mod

        def raise_url_error(req):
            raise urllib.error.URLError("conn reset")

        monkeypatch.setattr(mod.urllib.request, "urlopen", raise_url_error)
        collector = OpenAiStreamingCollector("http://fake", "Bearer x", "m")
        result = collector.measure([{"role": "user", "content": "hi"}])
        assert result["streaming_failed"] is True
        assert result["e2e_time_s"] is None
