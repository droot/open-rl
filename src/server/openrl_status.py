"""The model view: what the API server knows about one model, for clients
without kubectl (design 016, section 5.2).

Most clients run on a laptop behind a port-forward. For a model id the client
owns, this answers what `kubectl get workloads,pods` would: which workers the
model has, where each one is in its life, and whether its session is live.
Nothing here changes state. The worker manager answers the cluster side; the
stores answer the rest.

The view is scoped to one model on purpose. A model id is a random UUID the
server minted for the client that created it, so knowing the id is the
capability, as for every model-scoped Tinker call. Fleet-wide views wait for
a tenancy design.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from server.model_metadata import TrainingModelMetadata, get_model_metadata
from server.session_registry import SessionRegistry
from server.store import StateStore
from server.worker_manager import READY, STARTING, WorkerManager, owner_of

# What a worker entry exposes. The launchers know more (pod names, pids,
# owners); none of that is the client's contract.
WORKER_FIELDS = ("role", "state", "detail", "workload", "node", "age_seconds")


@dataclass
class StatusContext:
  state: StateStore
  registry: SessionRegistry
  manager: WorkerManager | None


def describe_worker(worker: dict[str, Any]) -> str:
  """One line for a human or an example's progress output."""
  age = f", {int(worker['age_seconds'])}s" if worker.get("age_seconds") is not None else ""
  return f"{worker['role']} {worker['state']}: {worker['detail']}{age}"


async def workers_of_owner(ctx: StatusContext, owner: str) -> list[dict[str, Any]]:
  if ctx.manager is None:
    return []
  out = []
  for found in await asyncio.to_thread(ctx.manager.inspect_owner, owner):
    worker = {field: found.get(field) for field in WORKER_FIELDS}
    # The launcher sees a sampler's pod run minutes before vLLM has loaded the
    # model; the sampler announces itself when it can serve.
    announced = found["role"] != "sampler" or await ctx.state.get_value(f"open_rl:sampler_ready:{found['model_id']}") is not None
    if worker["state"] == READY and not announced:
      worker["state"] = STARTING
      worker["detail"] = f"{worker['detail']}, loading the model"
    worker["summary"] = describe_worker(worker)
    out.append(worker)
  return out


async def owner_has_live_session(ctx: StatusContext, owner: str) -> bool:
  # Read-only: in_use() drops dead sessions, this only looks.
  for session_id in await ctx.state.set_members(f"open_rl:owner:{owner}"):
    if await ctx.registry.live(session_id):
      return True
  return False


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


async def model_status(ctx: StatusContext, model_id: str) -> dict[str, Any] | None:
  meta = await get_model_metadata(ctx.state, model_id)
  if meta is None:
    return None
  owner = await asyncio.to_thread(owner_of, model_id)
  workers = await workers_of_owner(ctx, owner)
  entry: dict[str, Any] = {
    "model_id": model_id,
    "base_model": meta.base_model,
    "fine_tuning_type": meta.fine_tuning_type,
    "status": meta.status,
    "created_at": meta.created_at,
    "shares_workers": meta.shares_runtime(),
    "session_live": await owner_has_live_session(ctx, owner),
    "settings": settings_of(meta),
    "workers": workers,
  }
  entry["summary"] = summarize(entry, workers)
  return entry
