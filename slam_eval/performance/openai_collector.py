"""OpenAI-compatible streaming collector: TTFT/TPOT/token counts (FR5)."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


class OpenAiStreamingCollector:
    """Performs a streaming chat-completion request and measures timings.

    First content chunk timestamps TTFT; inter-chunk deltas yield TPOT
    inputs. Server-reported usage (``stream_options.include_usage``) is
    preferred over chunk counting for token counts.
    """

    def __init__(self, url: str, authorization: Optional[str], model: str) -> None:
        self.url = url
        self.authorization = authorization
        self.model = model

    def measure(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: Optional[int] = None,
        non_streaming: bool = False,
    ) -> dict[str, Any]:
        """One request = one measurement (and the prediction itself).

        ``non_streaming`` (FR5 ``disabled_in_config``): plain request, no
        timing beyond e2e; FR3 metrics are null without a fallback warning.
        """
        if non_streaming:
            return self._measure_non_streaming(messages, max_output_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": self.authorization} if self.authorization else {}),
            },
        )
        t_start = time.perf_counter()
        ttft: Optional[float] = None
        t_last_chunk: Optional[float] = None
        chunk_count = 0
        content_parts: list[str] = []
        usage: Optional[dict[str, Any]] = None
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t_start
                        t_last_chunk = now - t_start  # elapsed, not absolute
                        chunk_count += 1
                        content_parts.append(content)
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                # Authorization failure is a configuration error, not a
                # server capability: fail loudly, never fall back silently.
                raise RuntimeError(
                    f"Authorization failed (HTTP {err.code}) for {self.url}. "
                    "Check the API key / credentials configuration."
                ) from err
            LOGGER.warning("Streaming request rejected: HTTP %s", err.code)
            return {
                "streaming_failed": True,
                "fallback_reason": "streaming_request_rejected",
            }
        except (urllib.error.URLError, OSError) as err:
            LOGGER.warning("Streaming request failed: %s", err)
            return {
                "streaming_failed": True,
                "fallback_reason": "streaming_request_rejected",
                "e2e_time_s": None,  # no completed request: no fabricated 0
            }
        e2e = time.perf_counter() - t_start
        if ttft is None or chunk_count == 0:
            return {
                "streaming_failed": True,
                "fallback_reason": "streaming_request_rejected",
            }
        if usage is not None:
            prompt_tokens = usage.get("prompt_tokens")
            generated_tokens = usage.get("completion_tokens")
            tokens_source = "usage"
        else:
            prompt_tokens = None
            generated_tokens = chunk_count  # approximate fallback
            tokens_source = "chunk_count"
        generated = generated_tokens if generated_tokens is not None else chunk_count
        tpot: Optional[float] = None
        if t_last_chunk is not None and generated >= 2 and ttft is not None:
            tpot = (t_last_chunk - ttft) / (generated - 1)
        return {
            "streaming_failed": False,
            "content": "".join(content_parts),
            "e2e_time_s": e2e,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "tokens_source": tokens_source,
        }

    def _measure_non_streaming(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: Optional[int],
    ) -> dict[str, Any]:
        import urllib.error

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": self.authorization} if self.authorization else {}),
            },
        )
        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                raise RuntimeError(
                    f"Authorization failed (HTTP {err.code}) for {self.url}. "
                    "Check the API key / credentials configuration."
                ) from err
            LOGGER.warning("Non-streaming request rejected: HTTP %s", err.code)
            return {
                "streaming_failed": True,
                "fallback_reason": "streaming_request_rejected",
            }
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as err:
            LOGGER.warning("Non-streaming request failed: %s", err)
            return {
                "streaming_failed": True,
                "fallback_reason": "streaming_request_rejected",
                "e2e_time_s": None,
            }
        e2e = time.perf_counter() - t_start
        choices = data.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
        usage = data.get("usage") or {}
        return {
            "streaming_failed": False,
            "content": content,
            "e2e_time_s": e2e,
            "ttft_s": None,
            "tpot_s": None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "tokens_source": "usage" if usage else None,
        }
