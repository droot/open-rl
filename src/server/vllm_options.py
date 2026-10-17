"""Engine and sampling options shared by the two vLLM sampler workers."""

import os
from collections.abc import Sequence
from typing import Any


def text_only_engine_kwargs() -> dict:
  """Stop vLLM reserving an encoder cache OpenRL can never use.

  Text checkpoints published under a `*ForConditionalGeneration` architecture
  make vLLM profile a max-resolution image and video at startup and reserve a
  multi-GiB encoder cache before the KV cache is sized, which can OOM engine
  init. OpenRL only ever passes token ids, so that capacity is unreachable.
  `VLLM_ENABLE_MULTIMODAL=1` restores stock vLLM behaviour.
  """
  if os.getenv("VLLM_ENABLE_MULTIMODAL", "0") == "1":
    return {}
  return {"limit_mm_per_prompt": {"image": 0, "video": 0}}


def split_stop(stop: str | Sequence[str] | Sequence[int] | None) -> tuple[list[str] | None, list[int] | None]:
  """Split a sampling request's `stop` into vLLM's `stop` and `stop_token_ids`.

  The sampling API types stop as `str | Sequence[str] | Sequence[int]` and which
  one arrives depends on the client's renderer. vLLM keeps strings and token ids
  in separate arguments and rejects a string in `stop_token_ids`. Each half is
  None when empty so callers do not override a vLLM default with [].
  """
  if stop is None:
    return None, None
  if isinstance(stop, str):
    return [stop], None
  strings = [s for s in stop if isinstance(s, str)]
  # bool is an int subclass; a stray True would be read as token id 1.
  token_ids = [t for t in stop if isinstance(t, int) and not isinstance(t, bool)]
  return (strings or None), (token_ids or None)


def gpu_memory_utilization() -> float:
  """vLLM's share of the device: an explicit VLLM_GPU_MEMORY_UTILIZATION, else 0.90.

  OPEN_RL_ACCELERATOR_MEMORY is deliberately not consulted. It is the placement
  claim the scheduler sized this worker for, not a runtime cap: workers that
  share an accelerator are time-sliced, so whichever one holds the device may
  use all of it. Sizing vLLM from the claim starved the sampler of KV cache on
  large devices and made Gemma-4's first engine init fail when the cold
  torch.compile transient did not fit in the claim.
  """
  explicit = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
  if explicit:
    return float(explicit)
  return 0.90


def sampler_batch_limits(device_memory_bytes: int | None = None) -> dict[str, int]:
  """vLLM's concurrency for this device: sequences in flight and prefill tokens per step.

  A sweep's sampler serves many groups at once, and the defaults (64
  sequences, 2048 prefill tokens) leave most of an 80 GB card idle. Explicit
  VLLM_MAX_NUM_SEQS / VLLM_MAX_NUM_BATCHED_TOKENS win; otherwise size by memory.
  """
  if device_memory_bytes is None:
    device_memory_bytes = detected_device_memory_bytes()
  gib = device_memory_bytes / 2**30
  if gib >= 60:
    seqs, tokens = 256, 16384
  elif gib >= 36:
    seqs, tokens = 128, 8192
  else:
    seqs, tokens = 64, 4096
  return {
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", seqs)),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", tokens)),
  }


def detected_device_memory_bytes() -> int:
  try:
    import torch

    if torch.cuda.is_available():
      return int(torch.cuda.get_device_properties(0).total_memory)
  except Exception:
    pass
  return 0


# Batch sizes vLLM records a CUDA graph for. Its default list has dozens of
# entries up to max_num_seqs and capturing them took 12-40 s on every sampler
# start (runs/2026-10-17_demo_baselines). A GRPO step sends a handful of
# shapes; powers of two up to max_num_seqs cover them with padding and cut the
# capture to a few sizes. VLLM_CUDAGRAPH_CAPTURE_SIZES: "default" keeps vLLM's
# list, otherwise a comma-separated list of sizes.
def cudagraph_capture_sizes(max_num_seqs: int) -> list[int] | None:
  raw = os.getenv("VLLM_CUDAGRAPH_CAPTURE_SIZES", "").strip().lower()
  if raw == "default":
    return None
  if raw:
    sizes = sorted({int(s) for s in raw.split(",") if s.strip()})
  else:
    sizes, size = [], 1
    while size <= max_num_seqs:
      sizes.append(size)
      size *= 2
  return [s for s in sizes if s <= max_num_seqs] or [max_num_seqs]


def compilation_kwargs(max_num_seqs: int) -> dict[str, Any]:
  sizes = cudagraph_capture_sizes(max_num_seqs)
  return {} if sizes is None else {"compilation_config": {"cudagraph_capture_sizes": sizes}}
