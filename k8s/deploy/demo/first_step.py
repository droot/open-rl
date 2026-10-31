"""Time to first step for a sweep, from the inspection API: start a sweep through the demo app, then watch every job
until its first optimizer step is recorded, and report where the time went.

  uv run --project src/server python k8s/deploy/demo/first_step.py --preset qwen-8 --steps 2 --label baseline

Milestones per job, all from records (no logs): the Job's creation, its pod's start, the session, the run
(create_model), the trainer's and sampler's placement, their first GPU turn, the first sample group, and the end of
step 0 (its optimizer step and the save after it). Writes <out>/<label>.json and prints a table with p50 and p90 per
kind. Needs the demo on --demo and the API server on --api (port-forwards will do).
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

MILESTONES = ("pod", "session", "run", "trainer_placed", "sampler_placed", "trainer_turn", "sampler_turn", "first_sample", "first_step")


def iso(stamp: float | None) -> str:
  return datetime.fromtimestamp(stamp, UTC).strftime("%H:%M:%S") if stamp else "-"


def pct(values: list[float], q: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class Watch:
  def __init__(self, demo: str, api: str):
    self.demo, self.api = httpx.Client(base_url=demo, timeout=60), httpx.Client(base_url=api, timeout=60)
    self.sessions: dict[str, float] = {}

  def start(self, run: str, jobs: list[dict]) -> list[str]:
    body = self.demo.post("/api/sweep/start", json={"run": run, "jobs": jobs}).raise_for_status().json()
    return [c["name"] for c in body["created"]]

  def jobs(self, run: str) -> list[dict]:
    return [j for j in self.demo.get("/api/sweep/jobs").raise_for_status().json()["jobs"] if j["run"] == run]

  def session_created(self, session_id: str) -> float | None:
    if session_id not in self.sessions:
      body = self.api.get(f"/api/v1/inspect/sessions/{session_id}/summary")
      if body.status_code != 200:
        return None
      self.sessions[session_id] = body.json().get("created_at")
    return self.sessions[session_id]

  def milestones(self, job: dict, run: dict | None) -> dict[str, float | None]:
    """What the records say so far for one job; None where the milestone has not happened or is not recorded yet."""
    found: dict[str, float | None] = dict.fromkeys(MILESTONES)
    found["pod"] = job.get("started_at")
    if not run:
      return found
    found["run"] = run.get("created_at")
    found["session"] = min((s for s in (self.session_created(sid) for sid in run.get("session_ids") or []) if s), default=None)
    since = int(found["run"] or time.time()) - 120
    lanes = self.api.get("/api/v1/inspect/activity", params={"run_id": run["run_id"], "since": since}).raise_for_status().json()["lanes"]
    born = job["created_at"]
    for lane in lanes:
      role = lane["role"]
      # a lane merges a Workload's placements; a shared runtime's lane can start before this job existed
      holds = [iv for iv in lane["intervals"] if iv[1] >= born] or lane["intervals"][-1:]
      if holds:
        found[f"{role}_placed"] = min(found[f"{role}_placed"] or 1e18, max(holds[0][0], born))
      turns = [t for t in lane["turns"] if t[1] >= born]
      mine = [a for a in lane.get("active", {}).get(run["run_id"], []) if a[1] >= born]
      if turns:
        found[f"{role}_turn"] = min(found[f"{role}_turn"] or 1e18, turns[0][0])
      elif mine:  # a resident worker records no turns; its first work for this run stands in
        found[f"{role}_turn"] = min(found[f"{role}_turn"] or 1e18, mine[0][0])
      if role == "sampler" and mine:
        found["first_sample"] = min(found["first_sample"] or 1e18, mine[0][1])
    steps = self.api.get(f"/api/v1/inspect/runs/{run['run_id']}/steps", params={"last": 3, "fields": "none"})
    if steps.status_code == 200:
      done = [s for s in steps.json()["steps"] if s["complete"]]
      if done:
        found["first_step"] = done[0]["ended_at"]
    return found


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--demo", default="http://127.0.0.1:8090")
  parser.add_argument("--api", default="http://127.0.0.1:8000")
  parser.add_argument("--preset", default="qwen-8", help="one of the demo's presets, or 'none' with --jobs")
  parser.add_argument("--jobs", help='JSON list of {"kind","model","count","steps","seed"} instead of a preset')
  parser.add_argument("--steps", type=int, default=2, help="steps per job (the sweep only needs the first)")
  parser.add_argument("--label", default=time.strftime("%H%M%S"))
  parser.add_argument("--run", help="sweep name; default bench-<label>")
  parser.add_argument("--attach", help="do not start anything: watch this existing sweep's jobs")
  parser.add_argument("--out", default=f"runs/{time.strftime('%Y-%m-%d')}_first_step")
  parser.add_argument("--deadline", type=float, default=1200, help="seconds to wait for every job's first step")
  parser.add_argument("--interval", type=float, default=5)
  args = parser.parse_args()

  watch = Watch(args.demo, args.api)
  if args.jobs:
    spec = json.loads(args.jobs)
  else:
    spec = [dict(j) for j in watch.demo.get("/api/sweep/presets").raise_for_status().json()["presets"][args.preset]["jobs"]]
  for j in spec:
    j["steps"] = args.steps
  if args.attach:
    run = args.attach
    names = [j["name"] for j in watch.jobs(run)]
    print(f"[first-step] watching {len(names)} jobs of sweep {run}", flush=True)
  else:
    run = args.run or f"bench-{args.label}"
    names = watch.start(run, spec)
    print(f"[first-step] started {len(names)} jobs as sweep {run}", flush=True)

  started, rows = time.time(), {}
  while time.time() - started < args.deadline:
    try:
      jobs = watch.jobs(run)
      inventory = watch.api.get("/api/v1/inspect/inventory").raise_for_status().json()
    except httpx.HTTPError as exc:  # a port-forward blip or a busy server; the records keep
      print(f"[first-step] read failed, retrying: {exc}", file=sys.stderr, flush=True)
      time.sleep(args.interval)
      continue
    runs = {r["run_id"]: r for r in inventory["runs"]}
    pending = 0
    for job in jobs:
      fresh = {"job": job["name"], "kind": job["kind"], "model": job["model"], "created": job["created_at"], "state": job["state"]}
      row = rows.setdefault(job["name"], fresh)
      row["state"] = job["state"]
      ended = job["state"] in ("succeeded", "failed")
      if ended:
        row.setdefault("ended_seen", time.time())
      # an ended job is still read for a while: its run id and last records can land after the Job completes
      if row.get("first_step") or (ended and time.time() - row["ended_seen"] > 90):
        continue
      try:
        row.update(watch.milestones(job, runs.get(job.get("run_id") or "")))
      except httpx.HTTPError as exc:  # one failed read is retried next round
        print(f"[first-step] {job['name']}: {exc}", file=sys.stderr, flush=True)
      if not row.get("first_step"):
        pending += 1
    done = sum(1 for r in rows.values() if r.get("first_step"))
    print(f"[first-step] +{time.time() - started:5.0f}s  first step done {done}/{len(names)}", flush=True)
    if pending == 0 and len(rows) == len(names):
      break
    time.sleep(args.interval)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  table = []
  for row in sorted(rows.values(), key=lambda r: r["job"]):
    t0 = row["created"]
    rel = {m: (round(row[m] - t0, 1) if row.get(m) else None) for m in MILESTONES}
    table.append({"job": row["job"], "kind": row["kind"], "model": row["model"], "created_at": iso(t0), "state": row["state"], **rel})
  (out / f"{args.label}.json").write_text(json.dumps({"sweep": run, "spec": spec, "jobs": table}, indent=1))

  head = ["job", "kind", "model", *MILESTONES]
  widths = [max(len(h), *(len(str(r.get(h, ""))) for r in table)) for h in head]
  print("\nseconds after the Job was created")
  print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
  for r in table:
    print("  ".join(str(r.get(h, "") if r.get(h) is not None else "-").ljust(w) for h, w in zip(head, widths)))
  print("\nfirst step, seconds after the Job was created")
  for kind in sorted({r["kind"] for r in table}):
    for model in sorted({r["model"] for r in table if r["kind"] == kind}):
      values = [r["first_step"] for r in table if r["kind"] == kind and r["model"] == model and r["first_step"] is not None]
      missing = sum(1 for r in table if r["kind"] == kind and r["model"] == model and r["first_step"] is None)
      low = min(values) if values else None
      print(f"  {kind:4} {model:22} n={len(values):2}  p50={pct(values, 0.5)}  p90={pct(values, 0.9)}  min={low}  missing={missing}")
  print(f"\nwritten {out / f'{args.label}.json'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
