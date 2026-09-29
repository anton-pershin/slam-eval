"""Memory sampling: background thread reading VRAM/RSS, phase-tagged (FR6/FR7)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via the mocked unit test
    import pynvml

    _HAS_PYNVML = True
except ImportError:  # pragma: no cover
    _HAS_PYNVML = False

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False
    psutil = None  # type: ignore[assignment]


class MemorySample(dict):
    """Phase-tagged memory sample: {timestamp, phase, vram_bytes, rss_bytes}."""


class MemorySampler:
    """Daemon thread sampling VRAM (per-PID, pynvml) and RSS (psutil).

    Samples are tagged with the monitor's current phase. Read-only: no torch
    or CUDA synchronization is performed. Failures degrade to warnings and
    null samples (NFR3) -- they never fail the evaluation.
    """

    def __init__(
        self,
        interval_s: float,
        phase_getter: Any,
        gpu_pids: Optional[list[int]] = None,
        ram_pids: Optional[list[int]] = None,
        device_index: int = 0,
    ) -> None:
        self._interval_s = interval_s
        self._phase_getter = phase_getter
        self._gpu_pids = gpu_pids or []
        self._ram_pids = ram_pids or []
        self._device_index = device_index
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_handles: dict[int, Any] = {}
        self._nvml_available = _HAS_PYNVML

    def start(self) -> None:
        if not self._ram_pids and not self._gpu_pids:
            LOGGER.warning("MemorySampler configured with no targets; not starting")
            return
        if self._gpu_pids and not self._nvml_available:
            LOGGER.warning("pynvml unavailable; VRAM will not be sampled")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            time.sleep(self._interval_s)

    def _sample_once(self) -> None:
        sample: dict[str, Any] = {
            "timestamp": time.time(),
            "phase": self._phase_getter(),
            "vram_bytes": None,
            "rss_bytes": None,
        }
        try:
            if self._gpu_pids and self._nvml_available:
                sample["vram_bytes"] = self._read_vram()
            if self._ram_pids and _HAS_PSUTIL:
                sample["rss_bytes"] = self._read_rss()
        except Exception:  # noqa: BLE001 - sampler must never crash the eval
            LOGGER.exception("Memory sampler failed once; sample may be partial")
        with self._lock:
            self._samples.append(sample)

    def _read_vram(self) -> int:
        import pynvml  # local import so blocking it for tests is easy

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            total = 0
            for pid in self._gpu_pids:
                try:
                    proc = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
                except AttributeError:  # older pynvml
                    proc = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in proc:
                    if p.pid == pid:
                        total += int(p.usedGpuMemory or 0)
            return total
        finally:
            pynvml.nvmlShutdown()

    def _read_rss(self) -> int:
        if psutil is None:
            return 0
        total = 0
        for pid in self._ram_pids:
            try:
                total += psutil.Process(pid).memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                LOGGER.warning("PID %s for RSS sampling unavailable", pid)
        return total

    def drain_samples(self) -> list[dict[str, Any]]:
        """Return and clear all samples collected so far."""
        with self._lock:
            samples = self._samples
            self._samples = []
        return samples
