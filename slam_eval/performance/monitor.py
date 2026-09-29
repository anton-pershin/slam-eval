"""Performance monitor: lifecycle hooks, collectors, aggregation (spec 04)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Optional

from slam_eval.performance.sampler import MemorySampler
from slam_eval.performance.stats import aggregate

LOGGER = logging.getLogger(__name__)

PHASE_PREDICT = "predict"
PHASE_SCORE = "score"
PHASE_IDLE = "idle"


class TokenTimingState:
    """Accumulates per-call token timing events (in-process backend)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t_start: Optional[float] = None
        self.t_first_token: Optional[float] = None
        self.t_last_token: Optional[float] = None
        self.generated_tokens = 0

    def on_step(self) -> None:
        now = time.perf_counter()
        if self.t_first_token is None:
            self.t_first_token = now
        self.t_last_token = now
        self.generated_tokens += 1


class PerformanceMonitor:
    """Collects performance metrics around the existing evaluation loop.

    The loop calls the lifecycle hooks at phase boundaries; a disabled or
    omitted monitor means the loop simply never calls any hook (FR1).
    """

    def __init__(
        self,
        stats: Optional[list[str]] = None,
        warmup_cases: int = 0,
        sampler_interval_s: float = 0.1,
        gpu_pids: Optional[list[int]] = None,
        ram_pids: Optional[list[int]] = None,
        device_index: int = 0,
        step_callback: Optional[Callable[[int], None]] = None,
        streaming: bool = True,
    ) -> None:
        self.stats = (
            stats
            if stats is not None
            else ["min", "max", "mean", "median", "q5", "q95"]
        )
        self.warmup_cases = warmup_cases
        self.streaming = streaming
        self._raw_records: list[dict[str, Any]] = []
        self._memory_samples: list[dict[str, Any]] = []
        self._phase = PHASE_IDLE
        self._t_prediction_start: Optional[float] = None
        self._streaming_state: dict[str, Any] = {
            "supported": True,
            "fallback_reason": None,
        }
        self._warned_streaming = False
        self._token_state = TokenTimingState()
        self._on_step_hook: Optional[Callable[[], None]] = None

        self.openai_collector: Any = None
        self.streaming_fallback_intended = False
        self._sampler: Optional[MemorySampler] = None
        if ram_pids or gpu_pids:
            self._sampler = MemorySampler(
                interval_s=sampler_interval_s,
                phase_getter=lambda: self._phase,
                gpu_pids=gpu_pids,
                ram_pids=ram_pids,
                device_index=device_index,
            )

    # ------------------------------------------------------------------
    # Public hooks called by the evaluation loop
    # ------------------------------------------------------------------
    def on_run_start(self, group_id: str) -> None:
        if self._sampler is not None:
            self._sampler.start()

    def on_prediction_start(self, case_index: int) -> None:
        self._phase = PHASE_PREDICT
        self._t_prediction_start = time.perf_counter()
        self._token_state.reset()
        self._token_state.t_start = self._t_prediction_start

    def on_prediction_end(self, case_index: int) -> None:
        t_end = time.perf_counter()
        e2e = (
            t_end - self._t_prediction_start
            if self._t_prediction_start is not None
            else None
        )
        record = self._build_record(case_index, e2e)
        self._raw_records.append(record)
        self._phase = PHASE_IDLE

    def on_scoring_start(self) -> None:
        self._phase = PHASE_SCORE

    def on_scoring_end(self) -> None:
        self._phase = PHASE_IDLE

    def on_run_end(self, group_id: str, run_metadata: dict[str, Any]) -> dict[str, Any]:
        """Aggregate; the caller saves via the storage adapter."""
        if self._sampler is not None:
            self._sampler.stop()
            self._memory_samples = self._sampler.drain_samples()
        return self.build_aggregated(group_id, run_metadata)

    # ------------------------------------------------------------------
    # Model-facing hooks
    # ------------------------------------------------------------------
    def make_step_callback(self) -> Callable[[int], None]:
        """Returns the callback to hand to LocalCausalLm(step_callback=...)."""

        def _on_step(token_id: int) -> None:
            self._token_state.on_step()

        return _on_step

    def note_usage_fallback(self) -> None:
        """Server did not report usage; token counts come from chunk counting."""
        self._streaming_state["fallback_reason"] = (
            "usage_not_reported_chunk_counting_used"
        )
        if not self._warned_streaming:
            LOGGER.warning(
                "Server usage not reported; token counts are approximate (chunk counting)"
            )
            self._warned_streaming = True

    def note_streaming_fallback(self, reason: str, warn: bool = True) -> None:
        self._streaming_state["supported"] = False
        self._streaming_state["fallback_reason"] = reason
        if warn and not self._warned_streaming:
            LOGGER.warning("Streaming unavailable: %s; TTFT/TPOT will be null", reason)
            self._warned_streaming = True

    # ------------------------------------------------------------------
    # Record construction / aggregation
    # ------------------------------------------------------------------
    def note_openai_result(
        self,
        case_index: int,
        e2e_s: float,
        ttft_s: Optional[float],
        generated_tokens: Optional[int],
        prompt_tokens: Optional[int],
    ) -> None:
        """Record a completed OpenAI-path call measured by the collector."""
        tpot = self._compute_tpot(ttft_s, e2e_s, generated_tokens)
        self._raw_records.append(
            {
                "case_id": case_index,
                "e2e_time_s": e2e_s,
                "ttft_s": ttft_s,
                "tpot_s": tpot,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "warmup": case_index < self.warmup_cases,
            }
        )

    def _build_record(self, case_index: int, e2e: Optional[float]) -> dict[str, Any]:
        ts = self._token_state
        ttft = (
            ts.t_first_token - ts.t_start
            if ts.t_first_token is not None and ts.t_start is not None
            else None
        )
        tpot = self._compute_tpot_from_state(ts, e2e)
        return {
            "case_id": case_index,
            "e2e_time_s": e2e,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "prompt_tokens": None,  # filled by the in-process collector path
            "generated_tokens": ts.generated_tokens if ts.generated_tokens else None,
            "warmup": case_index < self.warmup_cases,
        }

    @staticmethod
    def _compute_tpot(
        ttft_s: Optional[float], e2e_s: float, generated_tokens: Optional[int]
    ) -> Optional[float]:
        if ttft_s is None or generated_tokens is None or generated_tokens < 2:
            return None
        return (e2e_s - ttft_s) / (generated_tokens - 1)

    @staticmethod
    def _compute_tpot_from_state(
        ts: TokenTimingState, e2e: Optional[float]
    ) -> Optional[float]:
        if (
            ts.t_first_token is None
            or ts.t_last_token is None
            or ts.generated_tokens < 2
        ):
            return None
        return (ts.t_last_token - ts.t_first_token) / (ts.generated_tokens - 1)

    def set_prompt_tokens(self, case_index: int, prompt_tokens: int) -> None:
        if self._raw_records and self._raw_records[-1]["case_id"] == case_index:
            self._raw_records[-1]["prompt_tokens"] = prompt_tokens

    def build_aggregated(
        self, group_id: str, run_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        non_warmup = [r for r in self._raw_records if not r.get("warmup")]
        metrics = [
            "e2e_time_s",
            "ttft_s",
            "tpot_s",
            "prompt_tokens",
            "generated_tokens",
        ]
        aggregated: dict[str, Any] = {}
        for metric in metrics:
            values = [r.get(metric) for r in non_warmup]
            aggregated[metric] = aggregate(values, self.stats)
        # Memory: predict-phase samples only, run level (FR7)
        predict_samples = [
            s for s in self._memory_samples if s.get("phase") == PHASE_PREDICT
        ]
        for mem_metric in ("vram_bytes", "rss_bytes"):
            mem_values = [s.get(mem_metric) for s in predict_samples]
            aggregated[mem_metric] = aggregate(mem_values, self.stats)
        aggregated["memory_samples_raw"] = self._memory_samples
        aggregated["run_metadata"] = {
            **run_metadata,
            "warmup_cases_excluded": self.warmup_cases,
            "streaming": dict(self._streaming_state),
        }
        return aggregated

    @property
    def raw_records(self) -> list[dict[str, Any]]:
        return self._raw_records
