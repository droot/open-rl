"""Read-only views of what the API server knows, for clients without kubectl.

Most clients run on a laptop behind a port-forward. Everything
`kubectl get workloads,pods` would show about a model, these views show
too: which workers a model has, where each one is in scheduling and why,
whether its session is still live, and whether the fleet has room.

Nothing here changes state. The worker manager answers the cluster side
(design 016, section 5.2); the stores answer the rest.
"""

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from server.model_metadata import TrainingModelMetadata, get_model_metadata
from server.session_registry import SessionRegistry
from server.store import RequestStore, StateStore
from server.worker_manager import WorkerManager, owner_of

MODELS_SET = "open_rl:models"


@dataclass
class StatusContext:
  state: StateStore
  store: RequestStore
  registry: SessionRegistry
  manager: WorkerManager | None
  worker_launcher: str
  version: str


def describe_worker(worker: dict[str, Any]) -> str:
  """One line for a human or an example's progress output."""
  where = f" on {worker['node']}" if worker.get("node") else ""
  reason = f" ({worker['reason']})" if worker.get("reason") else ""
  pod = ""
  if worker.get("pod_waiting"):
    pod = f", pod {worker['pod_waiting']}"
  elif worker.get("pod_phase") and worker["pod_phase"] != "Running":
    pod = f", pod {worker['pod_phase']}"
  age = f", {int(worker['age_seconds'])}s" if worker.get("age_seconds") is not None else ""
  return f"{worker['role']} {worker.get('workload') or worker.get('runtime', '')} {worker['phase']}{reason}{where}{pod}{age}"


async def with_readiness(ctx: StatusContext, workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """A sampler is ready once it announced itself; a trainer once its pod runs."""
  out = []
  for worker in workers:
    worker = dict(worker)
    if worker.get("role") == "sampler" and worker.get("model_id"):
      worker["ready"] = await ctx.state.get_value(f"open_rl:sampler_ready:{worker['model_id']}") is not None
    else:
      worker["ready"] = worker.get("phase") == "Running" and worker.get("pod_phase", "Running") == "Running"
    worker["summary"] = describe_worker(worker)
    out.append(worker)
  return out


async def workers_of_owner(ctx: StatusContext, owner: str) -> list[dict[str, Any]]:
  if ctx.manager is None:
    return []
  return await with_readiness(ctx, await asyncio.to_thread(ctx.manager.inspect_owner, owner))


async def all_workers(ctx: StatusContext) -> list[dict[str, Any]]:
  if ctx.manager is None:
    return []
  return await with_readiness(ctx, await asyncio.to_thread(ctx.manager.inspect_all))


async def sessions_of_owner(ctx: StatusContext, owner: str) -> list[dict[str, Any]]:
  # Read-only: in_use() drops dead sessions, this only reports them.
  sessions = []
  for session_id in sorted(await ctx.state.set_members(f"open_rl:owner:{owner}")):
    sessions.append({"session_id": session_id, "live": await ctx.registry.live(session_id)})
  return sessions


async def cluster_status(ctx: StatusContext) -> dict[str, Any]:
  """What is running, who is using it, and whether the fleet has room."""
  store: dict[str, Any] = {"ok": True}
  owners: list[dict[str, Any]] = []
  try:
    for owner in await ctx.registry.owners():
      owners.append({"owner": owner, "sessions": await sessions_of_owner(ctx, owner)})
  except Exception as exc:
    store = {"ok": False, "error": str(exc)}
  workers = await all_workers(ctx)
  by_owner: dict[str, list[dict[str, Any]]] = {}
  for worker in workers:
    by_owner.setdefault(worker.get("owner") or "", []).append(worker)
  for entry in owners:
    entry["workers"] = by_owner.pop(entry["owner"], [])
  # Workers whose owner the registry forgot, or that another API server made.
  for owner, orphans in by_owner.items():
    owners.append({"owner": owner, "sessions": [], "workers": orphans})
  live = {s["session_id"] for entry in owners for s in entry["sessions"] if s["live"]}
  fleet = await asyncio.to_thread(ctx.manager.fleet) if ctx.manager is not None else None
  return {
    "version": ctx.version,
    "worker_launcher": ctx.worker_launcher,
    "store": store,
    # Only sessions attached to workers are enumerable; a session that never
    # created a model is not counted.
    "sessions": {"live": len(live)},
    "owners": owners,
    "workers_by_phase": dict(Counter(w["phase"] for w in workers)),
    "fleet": fleet,
  }


def settings_of(meta: TrainingModelMetadata) -> dict[str, Any]:
  """The model's settings as the client could have passed them."""
  return {
    "openrl.fine_tuning_type": meta.fine_tuning_type,
    "openrl.weight_sync": meta.weight_sync_config.strategy,
    "openrl.exclusive": meta.exclusive,
    "openrl.sampler_replicas": meta.sampler_replicas,
    "openrl.trainer_backend": meta.trainer_backend,
    "openrl.trainer_gpus": meta.trainer_gpus,
    "openrl.trainer_cp": meta.trainer_cp,
  }


async def model_entry(ctx: StatusContext, model_id: str, meta: TrainingModelMetadata, detail: bool) -> dict[str, Any]:
  owner = await asyncio.to_thread(owner_of, model_id)
  sessions = await sessions_of_owner(ctx, owner)
  entry: dict[str, Any] = {
    "model_id": model_id,
    "base_model": meta.base_model,
    "fine_tuning_type": meta.fine_tuning_type,
    "status": meta.status,
    "created_at": meta.created_at,
    "owner": owner,
    "shares_workers": meta.shares_runtime(),
    "session_live": any(s["live"] for s in sessions),
  }
  if detail:
    workers = await workers_of_owner(ctx, owner)
    entry["settings"] = settings_of(meta)
    entry["sessions"] = sessions
    entry["workers"] = workers
    entry["summary"] = summarize(entry, workers)
  return entry


def summarize(entry: dict[str, Any], workers: list[dict[str, Any]]) -> str:
  """What an agent should know first, in one line."""
  if entry["status"] != "active":
    return f"model {entry['status']}"
  if not workers:
    if entry["session_live"]:
      return "no workers yet; the API server has not created them or they were released"
    return "no live session and no workers; the model's workers were released"
  if not entry["session_live"]:
    ttl = "its session expired; the reaper releases unused workers within about two minutes"
    return f"{'; '.join(w['summary'] for w in workers)}. {ttl}"
  return "; ".join(w["summary"] for w in workers)


async def list_models(ctx: StatusContext) -> dict[str, Any]:
  models = []
  for model_id in sorted(await ctx.state.set_members(MODELS_SET)):
    meta = await get_model_metadata(ctx.state, model_id)
    if meta is not None:
      models.append(await model_entry(ctx, model_id, meta, detail=False))
  models.sort(key=lambda m: m["created_at"], reverse=True)
  return {"models": models}


async def model_status(ctx: StatusContext, model_id: str) -> dict[str, Any] | None:
  meta = await get_model_metadata(ctx.state, model_id)
  if meta is None:
    return None
  return await model_entry(ctx, model_id, meta, detail=True)


async def session_status(ctx: StatusContext, session_id: str) -> dict[str, Any] | None:
  live = await ctx.registry.live(session_id)
  owners = [owner for owner in await ctx.registry.owners() if session_id in await ctx.state.set_members(f"open_rl:owner:{owner}")]
  if not live and not owners:
    return None
  return {
    "session_id": session_id,
    "live": live,
    "ttl_seconds": ctx.registry.ttl_seconds,
    "user_metadata": await ctx.registry.user_metadata(session_id) if live else {},
    "owners": owners,
    "checked_at": time.time(),
  }
