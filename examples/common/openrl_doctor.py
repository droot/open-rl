"""Check an OpenRL deployment from the client's side, in under a minute, without a GPU.

  uv --project examples run openrl-doctor [--base-url http://127.0.0.1:9003] [--train]

Each check prints one line: OK, WARN or FAIL, then what was found. Exit code
is 1 if any check failed. --train runs a 3-step tiny_sft round trip and times
it, so you learn what "normal" takes on this cluster; it is the only check
that proves a worker can start.

Only the HTTP endpoint is used; no kubectl. Whether the GPU nodes have room
is not visible from the endpoint yet (design 016, deferred work).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:9003"))


class Report:
  def __init__(self) -> None:
    self.failed = False

  def line(self, level: str, check: str, detail: str) -> None:
    if level == "FAIL":
      self.failed = True
    print(f"{level:<4} {check:<14} {detail}", flush=True)


def get(base_url: str, path: str, timeout: float = 10.0) -> tuple[int, Any, float]:
  started = time.monotonic()
  try:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as response:
      body = response.read()
      return response.status, json.loads(body) if body else None, time.monotonic() - started
  except urllib.error.HTTPError as exc:
    return exc.code, None, time.monotonic() - started
  except Exception as exc:
    raise ConnectionError(f"{type(exc).__name__}: {exc}") from exc


def check_endpoint(report: Report, base_url: str) -> dict[str, Any] | None:
  """None when the server is unreachable or not OpenRL; {} when it is an
  OpenRL server too old to describe itself; else the openrl object."""
  check = "healthz"
  try:
    status, body, took = get(base_url, "/api/v1/healthz")
    if status != 200 or not body or body.get("status") != "ok":
      report.line("FAIL", check, f"HTTP {status} {body}")
      return None
    report.line("OK", check, f"{took * 1000:.0f} ms")

    check = "capabilities"
    status, caps, _ = get(base_url, "/api/v1/get_server_capabilities")
    if status != 200 or caps is None:
      report.line("FAIL", check, f"HTTP {status}")
      return None
    info = caps.get("openrl")
    if not info:
      report.line("WARN", check, "no openrl section: this server predates v0.0.7 and does not describe itself")
      return {}
    settings = ", ".join(name.removeprefix("openrl.") for name in info.get("settings", {}).get("properties", {}))
    report.line(
      "OK",
      "version",
      f"OpenRL {info.get('version', '?')}, workers via {info.get('worker_launcher', '?')}, "
      f"sampler {info.get('sampler_backend', '?')}, fft {info.get('fft_enabled', '?')}",
    )
    report.line("OK", "settings", settings or "none")
    limits = {k: v for k, v in info.get("limits", {}).items() if v is not None}
    report.line("OK", "limits", ", ".join(f"{k}={v}" for k, v in limits.items()))
    if info.get("skill_url"):
      report.line("OK", "skill", info["skill_url"])
    check = "openapi"
    status, _, _ = get(base_url, "/openapi.json")
    report.line("OK" if status == 200 else "WARN", check, f"/openapi.json HTTP {status}")
    return info
  except ConnectionError as exc:
    report.line("FAIL", check, f"cannot reach {base_url}: {exc}. Is the port-forward running?")
    return None


def check_train(report: Report, base_url: str) -> None:
  script = Path(__file__).resolve().parents[1] / "tiny" / "tiny_sft.py"
  cmd = [sys.executable, str(script), f"base_url={base_url}", "steps=3", "min_loss_drop=0.0", "sample_after_train=true"]
  started = time.monotonic()
  # Stream the example's own lines as they come, so the loss lines show up
  # while it runs rather than all at the end.
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  lines: list[str] = []
  last = "no output"
  assert proc.stdout is not None
  for raw in proc.stdout:
    line = raw.rstrip()
    if line:
      last = line
    if line.startswith("[tiny-sft]"):
      lines.append(line)
      print(f"     train          {line}", flush=True)
  proc.wait()
  took = time.monotonic() - started
  if proc.returncode != 0:
    report.line("FAIL", "train", f"tiny_sft exited {proc.returncode} after {took:.0f}s: {last}")
    return
  sampled = next((line.split("=", 1)[1] for line in lines if "sampled_saved_adapter" in line), "?")
  report.line("OK", "train", f"3 SFT steps plus one sample in {took:.0f}s; the adapter answered {sampled}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
  parser.add_argument("--train", action="store_true", help="run a 3-step tiny_sft round trip and time it")
  args = parser.parse_args()
  base_url = args.base_url.rstrip("/")

  report = Report()
  info = check_endpoint(report, base_url)
  if args.train and info is not None:
    check_train(report, base_url)
  sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
  main()
