"""Worker manager that creates one Workload per worker process and lets the
scheduler place it. Part of the cluster extra: importing it assumes the
Kubernetes client is installed.

Two FFT jobs and two LoRA jobs on the same base model come in:

  model_id (job)   owner               Workload (process)          shares
  fft-job 1a2b     1a2b                fft-1a2b-trainer            nothing
  fft-job 3c4d     3c4d                fft-3c4d-trainer            nothing
  lora-job 5e6f    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   one runtime
  lora-job 7a8b    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   with 5e6f
  automodel 9c0d   automodel-qwen-qwen2-5-0-5b   lora-automodel-qwen-qwen2-5-0-5b-0-trainer   one Automodel
                                                                                  runtime, apart from PyTorch's
  image 1d2e       image-<hash>-qwen-qwen2-5-0-5b   lora-image-<hash>-qwen-qwen2-5-0-5b-0-trainer   one runtime
                                                                                  per trainer image

The Workload name is what the pod label and the time-slicer call job_id.
"""

import dataclasses
import datetime
import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any

from kubernetes import client, config

from server.estimator import Footprint, footprint
from server.worker_manager import base_model_of, owner_id, runtime_of, worker_args, worker_env, worker_module

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"


def workload_name(role: str, owner: str, is_lora: bool, index: int = 0) -> str:
  # A second compatible request renders the same name, and the create's
  # AlreadyExists is the reuse. The index numbers sampler replicas; the first
  # keeps the name it always had.
  if is_lora:
    return f"lora-{owner}-{index}-{role}"
  return f"fft-{owner}-{role}" if index == 0 else f"fft-{owner}-{role}-{index}"


@dataclass(frozen=True)
class Worker:
  """Everything the API server knows about one worker process before placement."""

  role: str
  runtime: str
  base_model: str
  is_lora: bool
  exclusive: bool
  meta: Any
  footprint: Footprint
  # GPUs the worker drives as one torchrun group.
  devices: int = 1

  # Which replica of its role this is.
  index: int = 0

  @property
  def owner(self) -> str:
    return owner_id(self.runtime)

  @property
  def name(self) -> str:
    return workload_name(self.role, self.owner, self.is_lora, self.index)


def replicas_of(model_id: str, role: str) -> int:
  """How many workers of this role the model asked for. Each sampler replica
  is its own Workload draining the shared sampling queue."""
  if role != "sampler":
    return 1
  meta, _, _ = runtime_of(model_id)
  return meta.sampler_replicas


def describe_worker(model_id: str, role: str) -> Worker:
  meta, runtime, is_lora = runtime_of(model_id)
  base_model = base_model_of(meta, runtime)
  exclusive = not meta.shares_gpu()
  devices = meta.trainer_gpus if role == "trainer" else 1
  size = footprint(base_model, meta.fine_tuning_type, role)
  # Each torchrun rank is its own process with its own host memory.
  size = replace(size, host_request_bytes=size.host_request_bytes * devices, host_limit_bytes=size.host_limit_bytes * devices)
  return Worker(role, runtime, base_model, is_lora, exclusive, meta, size, devices)


def pod_env(worker: Worker) -> list[dict[str, Any]]:
  """The shared worker env plus what only the cluster knows. The time-slice
  group is placement's and the scheduler stamps it."""
  tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl")
  values = {
    "REDIS_URL": os.environ["REDIS_URL"],
    "OPEN_RL_TMP_DIR": tmp_dir,
    "HF_HOME": os.getenv("HF_HOME", f"{tmp_dir}/huggingface"),
    **worker_env(worker.meta, worker.base_model, worker.runtime, worker.is_lora, worker.role),
    "OPEN_RL_WORKLOAD_ID": worker.name,
    # The llmd snapshot agent still discovers processes by the older name.
    "OPEN_RL_TIME_SLICE_JOB_ID": worker.name,
    "OPEN_RL_ACCEL_TIMESLICER_PORT": os.getenv("OPEN_RL_ACCEL_TIMESLICER_PORT", "9753"),
  }
  # MAX_JOBS caps FlashInfer's JIT build, which otherwise runs one ~3GB
  # compiler per core and blows through the pod's host memory limit.
  for name in ("VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_MODEL_LEN", "OPEN_RL_TRAIN_TOKEN_BUDGET", "MAX_JOBS"):
    if os.getenv(name):
      values[name] = os.environ[name]
  # No other worker shares an exclusive worker's GPUs, so it never parks.
  if worker.exclusive:
    values["OPEN_RL_TIME_SLICING"] = "off"
  env: list[dict[str, Any]] = [{"name": name, "value": value} for name, value in values.items()]
  env.append({"name": "OPEN_RL_ACCEL_TIMESLICER_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}})
  return env


def worker_container(worker: Worker) -> tuple[str, list[str]]:
  """The image and command. An Automodel trainer, or one from an image the job
  names, runs the python on its image's PATH."""
  if worker.role == "trainer" and worker.meta.trainer_backend != "pytorch":
    image = worker.meta.trainer_image() or os.getenv("OPEN_RL_AUTOMODEL_IMAGE", "ghcr.io/gke-labs/open-rl/automodel:latest")
    if worker.devices > 1:
      torchrun = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={worker.devices}"]
      return image, ["python", "-u", *torchrun, "-m", worker_module(worker.role)]
    return image, ["python", "-u", "-m", worker_module(worker.role)]
  image = os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest")
  return image, ["uv", "run", "python", "-u", "-m", worker_module(worker.role)]


def pod_template(worker: Worker) -> dict[str, Any]:
  """The complete worker pod minus placement. Node selection and claims are
  the scheduler's; it rejects a template that carries them."""
  image, command = worker_container(worker)
  template = {
    "spec": {
      "restartPolicy": "OnFailure",
      "containers": [
        {
          "name": "worker",
          "image": image,
          "command": command,
          "args": worker_args(worker.runtime, worker.role, worker.is_lora),
          "env": pod_env(worker),
          "resources": worker.footprint.resources,
          "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
        }
      ],
      "volumes": [
        {
          "name": "shared-storage",
          "persistentVolumeClaim": {"claimName": os.getenv("OPEN_RL_SHARED_PVC", "open-rl-shared-pvc")},
        }
      ],
      "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    },
  }

  # NCCL moves data between ranks through /dev/shm, which is 64Mi by default.
  if worker.devices > 1:
    template["spec"]["containers"][0]["volumeMounts"].append({"name": "dshm", "mountPath": "/dev/shm"})
    template["spec"]["volumes"].append({"name": "dshm", "emptyDir": {"medium": "Memory"}})

  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    template["spec"]["containers"][0]["imagePullPolicy"] = pull_policy
  return template


def accelerator_spec(worker: Worker) -> dict[str, Any]:
  """A torchrun group asks for its devices, each sized for a whole replica."""
  if worker.devices > 1:
    return {"mode": "MultiGPU", "devices": worker.devices, "memory": worker.footprint.accelerator}
  return {"mode": "SingleGPU", "memory": worker.footprint.accelerator}


def workload_body(worker: Worker) -> dict[str, Any]:
  return {
    "apiVersion": f"{GROUP}/{VERSION}",
    "kind": "Workload",
    "metadata": {"name": worker.name, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}},
    "spec": {
      "role": worker.role,
      "trainingKind": "lora" if worker.is_lora else "fft",
      "exclusive": worker.exclusive,
      "modelID": worker.runtime,
      "ownerID": worker.owner,
      "accelerator": accelerator_spec(worker),
      "workerContainerName": "worker",
      "template": pod_template(worker),
    },
  }


class SchedulerWorkerManager:
  """Runs trainer and sampler workers by creating Workload objects."""

  def __init__(self, custom_api: Any = None, core_api: Any = None, resource_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    if custom_api is None:
      config.load_incluster_config()
      custom_api = client.CustomObjectsApi()
      core_api = client.CoreV1Api()
      resource_api = client.ResourceV1Api()
    self.custom_api = custom_api
    self.core_api = core_api
    self.resource_api = resource_api
    # Status views may be polled; one read per few seconds is plenty.
    self._view_cache: dict[str, tuple[float, Any]] = {}
    self.view_cache_seconds = 5.0

  def ensure(self, model_id: str, role: str) -> None:
    for index in range(replicas_of(model_id, role)):
      self.ensure_workload(dataclasses.replace(describe_worker(model_id, role), index=index))

  def ensure_workload(self, worker: Worker) -> None:
    role = worker.role
    deadline = time.monotonic() + 180
    while True:
      try:
        self.custom_api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, workload_body(worker))
        logger.info("requested %s workload %s (%s, owner %s)", role, worker.name, worker.footprint.accelerator, worker.owner)
        return
      except Exception as exc:
        if getattr(exc, "status", None) != 409:
          raise
      # AlreadyExists is the reuse, unless the old one is still being deleted.
      if not self.still_deleting(worker.name):
        return
      if time.monotonic() > deadline:
        raise RuntimeError(f"workload {worker.name} has been terminating for over three minutes")
      time.sleep(2)

  def still_deleting(self, name: str) -> bool:
    try:
      workload = self.custom_api.get_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return True  # gone since the create failed, so the next create will go through
      raise
    return bool(workload["metadata"].get("deletionTimestamp"))

  def release(self, model_id: str) -> None:
    try:
      meta, runtime, _ = runtime_of(model_id)
      shared = meta.shares_runtime()
    except Exception:
      runtime, shared = model_id, False
    if shared:
      return  # a shared runtime outlives any one job
    self.release_owner(owner_id(runtime))

  def release_owner(self, owner: str) -> set[str]:
    """Delete the owner's workloads. The scheduler's finalizer frees the seats."""
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    ours = [item for item in found["items"] if item["spec"]["ownerID"] == owner]
    for item in ours:
      self.delete_workload(item["metadata"]["name"])
    return {item["spec"]["modelID"] for item in ours}

  def close(self) -> None:
    pass  # Workloads outlive the API server; the scheduler owns them from here

  def delete_workload(self, name: str) -> None:
    try:
      self.custom_api.delete_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise

  # -- read-only views (design 016) -----------------------------------------

  def cached(self, key: str, load: Any) -> Any:
    now = time.monotonic()
    hit = self._view_cache.get(key)
    if hit is not None and now - hit[0] < self.view_cache_seconds:
      return hit[1]
    value = load()
    self._view_cache[key] = (now, value)
    return value

  def inspect_owner(self, owner: str) -> list[dict[str, Any]]:
    return [worker for worker in self.inspect_all() if worker["owner"] == owner]

  def inspect_all(self) -> list[dict[str, Any]]:
    """Our Workloads joined with their pods: what kubectl get workloads,pods
    shows, as one entry per worker."""
    return self.cached("workers", self.load_workers)

  def load_workers(self) -> list[dict[str, Any]]:
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    pods = {pod.metadata.name: pod for pod in self.core_api.list_namespaced_pod(self.namespace).items}
    now = time.time()
    workers = []
    for item in found["items"]:
      spec, status, meta = item["spec"], item.get("status") or {}, item["metadata"]
      worker: dict[str, Any] = {
        "workload": meta["name"],
        "role": spec["role"],
        "owner": spec.get("ownerID"),
        "model_id": spec["modelID"],
        "phase": status.get("phase") or "Pending",
        "reason": status.get("reason") or "",
        "node": status.get("nodeName"),
        "pod": status.get("podName"),
        "devices": status.get("deviceCount"),
        "memory_per_device": status.get("memoryPerDevice"),
        "age_seconds": age_seconds(meta.get("creationTimestamp"), now),
        "deleting": bool(meta.get("deletionTimestamp")),
      }
      pod = pods.get(worker["pod"]) if worker["pod"] else None
      if pod is not None:
        worker["pod_phase"] = pod.status.phase
        waiting = [c.state.waiting.reason for c in pod.status.container_statuses or [] if c.state and c.state.waiting]
        worker["pod_waiting"] = waiting[0] if waiting else None
        worker["restarts"] = sum(c.restart_count for c in pod.status.container_statuses or [])
      workers.append(worker)
    return workers

  def fleet(self) -> dict[str, Any] | None:
    """Accelerators on nodes the scheduler may use, from the DRA driver's
    ResourceSlices, and how many of them run an OpenRL worker. The API server
    cannot read ResourceClaims, so occupancy counts our Running workers, not
    every claim on the node."""
    try:
      return self.cached("fleet", self.load_fleet)
    except Exception as exc:
      return {"error": f"{type(exc).__name__}: {exc}"}

  def load_fleet(self) -> dict[str, Any]:
    nodes = {}
    for node in self.core_api.list_node(label_selector="openrl.io/enabled=true").items:
      labels = node.metadata.labels or {}
      # A node naming neither role accepts both; otherwise only the roles set to true.
      named = {role for role in ("trainer", "sampler") if f"openrl.io/{role}" in labels}
      roles = [role for role in ("trainer", "sampler") if not named or labels.get(f"openrl.io/{role}") == "true"]
      nodes[node.metadata.name] = {"name": node.metadata.name, "roles": roles, "accelerator": None, "memory": None, "devices": 0, "openrl_workers": 0}
    for s in self.resource_api.list_resource_slice().items:
      if s.spec.driver != "gpu.nvidia.com" or s.spec.node_name not in nodes:
        continue
      entry = nodes[s.spec.node_name]
      for device in s.spec.devices or []:
        attrs = device.attributes or {}
        if "productName" in attrs:
          entry["accelerator"] = attrs["productName"].string
        if device.capacity and "memory" in device.capacity:
          entry["memory"] = device.capacity["memory"].value
        entry["devices"] += 1
    for worker in self.inspect_all():
      if worker["phase"] == "Running" and worker["node"] in nodes:
        nodes[worker["node"]]["openrl_workers"] += worker.get("devices") or 1
    accelerators: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for entry in nodes.values():
      key = (entry["accelerator"], entry["memory"])
      acc = accelerators.setdefault(key, {"accelerator": key[0], "memory": key[1], "nodes": 0, "devices": 0, "openrl_workers": 0})
      acc["nodes"] += 1
      acc["devices"] += entry["devices"]
      acc["openrl_workers"] += entry["openrl_workers"]
    return {"nodes": sorted(nodes.values(), key=lambda n: n["name"]), "accelerators": list(accelerators.values())}


def age_seconds(timestamp: str | None, now: float) -> float | None:
  if not timestamp:
    return None
  created = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
  return max(0.0, now - created)
