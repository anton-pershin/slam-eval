# Performance Monitoring: Hosting-Specific Caveats

Root-cause notes from validating the performance monitor against real backends
(spec 04, AC10/AC11). Read this before interpreting TTFT/TPOT numbers or
debugging missing measurements.

## Caila (remote API): reasoning models and TTFT/TPOT coverage

**Symptom.** A monitored run against `glm-5.3-flash` on Caila reports
`fallback_reason: "streaming_request_rejected"` and TTFT/TPOT are null for a
subset of cases (all of them at small token budgets), even though the API key
is valid and a direct streaming probe returns HTTP 200.

**Root cause.** GLM-5.3-flash is a reasoning model: it emits internal
"thinking" tokens before the answer. Those thinking tokens consume the
`max_output_tokens` budget. When the budget is exhausted by reasoning, the API
sends a single empty chunk with `finish_reason: "length"` — zero content
chunks. The collector correctly reports "no streaming content" and the case
falls back to null metrics.

**Evidence** (single-case probes, real merge-quality prompt, 2026-09-30):

- `max_tokens=64` → usage: 55 reasoning tokens, 0 content chunks streamed
- `max_tokens=256` → 113 content chunks, TTFT 1.88 s, reasoning 49 tokens
- `max_tokens=1024` → 597 reasoning tokens (longer budget, more thinking)

**Consequences for measurement:**

1. For reasoning models via Caila, set `max_output_tokens` well above the
   reasoning budget (e.g. 1024) if you need per-case TTFT/TPOT for (almost)
   every case. With a tight budget, only prompts whose reasoning fits the
   budget produce streaming metrics; the rest are honestly null.
2. `enable_thinking: false` via `chat_template_kwargs` is **not** honored by
   the Caila OpenAI-compatible endpoint (verified: reasoning tokens unchanged
   with the flag), so the budget is the only lever there.
3. Prompt-dependent reasoning length also means TPOT/TTFT distributions have
   heavier tails than for non-reasoning models — compare medians, not means.

Related failure mode — **authorization**: a missing/invalid API key surfaces as
an HTTP 401/403, which the collector converts to
`RuntimeError("Authorization failed (HTTP …)")` and the eval stops. It is
never treated as a streaming-capability fallback. Separately, check that
`CAILA_API_KEY` is actually exported in the environment that runs the eval:
`.bashrc` is not read by non-interactive SSH commands, so the variable can be
absent precisely in scripted runs.

## Local vLLM: flashinfer JIT vs system CUDA toolkit

**Symptom.** `vllm serve` fails at engine startup with
`nvcc fatal: Unknown option '--compress-mode=size'` from a flashinfer ninja
build, even though torch works fine.

**Root cause.** vLLM's flashinfer integration JIT-compiles sampling kernels
with the system `nvcc` and passes a flag introduced in CUDA ≥ 12.4. Hosts with
an older system toolkit (e.g. CUDA 12.0) fail even when a newer toolkit exists
elsewhere on disk.

**Workaround.** `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve …` uses vLLM's
native sampler and skips the JIT path. This does not affect the performance
measurement methodology (sampling kernel choice is irrelevant to TTFT/TPOT
methodology). Long-term fix: put a CUDA ≥ 12.4 `nvcc` on PATH.

**Serving note.** vLLM streams genuine per-token chunks, so client-side TPOT
is trustworthy; VRAM attribution should point at the `VLLM::EngineCore` worker
PID (the API process holds almost no GPU memory), e.g. via
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`.

## In-process LocalCausalLm

Measured in the eval process itself (`self_monitor: true`): per-step callbacks
give TTFT/TPOT directly, and the process PID is used for RSS/VRAM sampling. No
known caveats beyond warmup handling (excluded via the `warmup` flag) and the
scoring phase being tagged separately so memory aggregation stays
predict-phase-only.
