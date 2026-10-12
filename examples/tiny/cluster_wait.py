"""Say what the cluster is doing while a Tinker call blocks.

The SDK blocks inside create_lora_training_client() and .result() while the
API server starts workers, which on a cold node takes minutes. Without this
the example prints nothing for that long and looks hung. A thread polls the
API server's status view and prints one line per minute, for example:

  [tiny-sft] waiting 2m0s on create_model; trainer lora-google-gemma-4-e2b-0-trainer Placing (PodCreated) on gke-...-lvw8, pod ContainerCreating, 118s

The status view is OpenRL's; against another Tinker server the thread stays quiet.
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


def cluster_summary(base_url: str) -> str:
  status = fetch_json(f"{base_url}/api/v1/openrl/status")
  if status is None:
    return "status view unavailable"
  lines = [worker["summary"] for owner in status.get("owners", []) for worker in owner.get("workers", [])]
  if lines:
    return "; ".join(lines)
  fleet = status.get("fleet") or {}
  if fleet.get("accelerators"):
    free = sum(a["devices"] - a["openrl_workers"] for a in fleet["accelerators"])
    return f"no workers yet; {free} free accelerator(s) in the fleet"
  return "no workers yet"


@contextmanager
def waiting(tag: str, base_url: str, what: str, after: float = 60.0, every: float = 60.0):
  """Print the cluster's view every `every` seconds once a call has blocked for `after`."""
  done = threading.Event()
  started = time.monotonic()

  def watch() -> None:
    if done.wait(after):
      return
    while not done.is_set():
      elapsed = int(time.monotonic() - started)
      print(f"[{tag}] waiting {elapsed // 60}m{elapsed % 60:02d}s on {what}; {cluster_summary(base_url)}")
      if done.wait(every):
        return

  thread = threading.Thread(target=watch, daemon=True)
  thread.start()
  try:
    yield
  finally:
    done.set()
