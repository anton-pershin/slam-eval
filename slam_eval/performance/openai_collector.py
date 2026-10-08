"""The measured OpenAI-compatible request, taken through the model's `Llm`.

Measuring is this module's job: time to first token, time per output token and
the token counts. Everything the request is made of — the endpoint, the
headers, the field names, the streaming framing, the server's end-of-stream
marker and its failure vocabulary — belongs to `rally.llm`, so the request
measured here is the request the model itself would send.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from rally.llm import (
    Llm,
    LlmAuthorizationError,
    LlmError,
    LlmStreamRejectedError,
    LlmUsage,
)

LOGGER = logging.getLogger(__name__)


class OpenAiStreamingCollector:
    """Measures one prediction through the `Llm` it is given.

    `measure` answers a plain dict whose keys are always present:
    `streaming_failed` says whether the measurement is a fallback record,
    `fallback_reason` names why, `content` carries the answer.
    """

    def __init__(self, llm: Llm) -> None:
        self.llm = llm

    def measure(
        self,
        messages: list[dict[str, str]],
        non_streaming: bool = False,
    ) -> dict[str, Any]:
        """Measure one request for `messages` through the Llm.

        `non_streaming` asks for the Llm's own non-streaming request; every
        timing metric is null on that path (spec 04 FR2).
        """
        if non_streaming:
            return self._measure_non_streaming(messages)

        return self._measure_streaming(messages)

    @staticmethod
    def _failed(reason: Optional[str], e2e_s: Optional[float] = None) -> dict[str, Any]:
        return {
            "streaming_failed": True,
            "fallback_reason": reason,
            "content": None,
            "e2e_time_s": e2e_s,
        }

    def _measure_non_streaming(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        start = time.perf_counter()
        message = self.llm.request(messages)
        e2e_s = time.perf_counter() - start

        if message is None:
            # The Llm's non-streaming request answers None for every failure —
            # rejected, dropped and timed out alike — so this record says only
            # that the measurement fell back: no reason, no timing (NFR4). The
            # loop below leaves the run's own reason (a config-intended one
            # included) untouched when no reason is named.
            LOGGER.warning(
                "Non-streaming request to %s produced no response; "
                "the record keeps no timing and no reason",
                self.llm.url,
            )
            return self._failed(None)

        return {
            "streaming_failed": False,
            "fallback_reason": None,
            "content": message.get("content") or "",
            "e2e_time_s": e2e_s,
            "ttft_s": None,
            "tpot_s": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "tokens_source": None,
        }

    def _measure_streaming(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        start = time.perf_counter()
        ttft_s: Optional[float] = None
        last_content_s: Optional[float] = None
        content_events = 0
        parts: list[str] = []
        usage: Optional[LlmUsage] = None
        truncated = False

        try:
            for event in self.llm.stream(messages):
                if event.usage is not None:
                    usage = event.usage
                if event.truncated:
                    truncated = True
                if event.content:
                    elapsed = time.perf_counter() - start
                    if ttft_s is None:
                        ttft_s = elapsed
                    last_content_s = elapsed
                    content_events += 1
                    parts.append(event.content)
        except LlmAuthorizationError as err:
            # Never a silent metrics fallback: broken credentials abort the run.
            raise RuntimeError(
                f"Authorization failed for {self.llm.url}. "
                "Check the API key / credentials configuration."
            ) from err
        except LlmStreamRejectedError as err:
            LOGGER.warning("Streaming request rejected: %s", err)
            return self._failed("streaming_request_rejected")
        except LlmError as err:
            LOGGER.warning("Streaming request failed: %s", err)
            return self._failed("streaming_request_rejected")

        e2e_s = time.perf_counter() - start

        if ttft_s is None and not truncated:
            # No content event arrived and the server did not declare the cap
            # cut the answer short: there is nothing to measure.
            return self._failed("streaming_request_rejected")

        if usage is not None:
            prompt_tokens: Optional[int] = usage.prompt_tokens
            generated_tokens: Optional[int] = usage.completion_tokens
            tokens_source: Optional[str] = "usage"
        else:
            prompt_tokens = None
            generated_tokens = content_events
            tokens_source = "chunk_count"

        counted = generated_tokens if generated_tokens is not None else content_events
        tpot_s: Optional[float] = None
        if ttft_s is not None and last_content_s is not None and counted >= 2:
            tpot_s = (last_content_s - ttft_s) / (counted - 1)

        return {
            "streaming_failed": False,
            "fallback_reason": None,
            "content": "".join(parts),
            "e2e_time_s": e2e_s,
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "tokens_source": tokens_source,
        }
