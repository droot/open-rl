"""Say what the cluster is doing while a Tinker call blocks.

The SDK blocks inside create_lora_training_client() and .result() while the
API server starts workers, which on a cold node takes minutes. Without this
the example prints nothing for that long and looks hung. A thread prints one
line per minute. Once the example knows its model id, the line carries the
model view's summary, for example:

  [tiny-sft] waiting 2m00s on the first sample; trainer ready: Running on gke-...-lvw8, 201s;
  sampler starting: Placing on gke-...-2208, pod ContainerCreating, 118s

During create_model the client has no model id yet, so the line only says
how long it has waited. The model view is OpenRL's; against another Tinker
server the line stays plain.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from typing import Any

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Everything an example prints is progress; make it show up when piped.
sys.stdout.reconfigure(line_buffering=True)


def fetch_json(url: str, timeout: float = 5.0) -> dict[str, Any] | None:
  try:
    with urllib.request.urlopen(url, timeout=timeout) as response:
      return json.loads(response.read())
  except Exception:
    return None


def model_summary(base_url: str, model_id: str) -> str:
  body = fetch_json(f"{base_url}/api/v1/openrl/models/{model_id}")
  return body.get("summary", "model view unavailable") if body else "model view unavailable"


@contextmanager
def waiting(tag: str, base_url: str, what: str, model_id: str | None = None, after: float = 60.0, every: float = 60.0):
  """Print one line every `every` seconds once a call has blocked for `after`;
  with a `model_id`, the line carries the model view's summary."""
  done = threading.Event()
  started = time.monotonic()

  def watch() -> None:
    if done.wait(after):
      return
    while not done.is_set():
      elapsed = int(time.monotonic() - started)
      line = f"[{tag}] waiting {elapsed // 60}m{elapsed % 60:02d}s on {what}"
      if model_id:
        line += f"; {model_summary(base_url, model_id)}"
      print(line)
      if done.wait(every):
        return

  thread = threading.Thread(target=watch, daemon=True)
  thread.start()
  try:
    yield
  finally:
    done.set()
