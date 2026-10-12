"""The read-only views of design 016: capabilities, status, models, sessions.

The ASGI client runs the real routes; the worker manager is the fake from the
session lifecycle tests, so every entry here reflects what a client behind a
port-forward would see, without a cluster."""

import asyncio
import os
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from server import api_server
from server.scheduler_worker_manager import SchedulerWorkerManager
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore
from tests.api_client import asgi_client, post_json
from tests.test_scheduler_worker_manager import FakeCustomObjectsApi
from tests.test_session_lifecycle import RuntimeManager


class StatusViewsTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.store = InMemoryStore()
    self.state = InMemoryStateStore()
    self.manager = RuntimeManager()
    self.registry = SessionRegistry(self.state)
    self.enterContext(patch.object(api_server, "state", self.state))
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "get_store", return_value=self.store))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.state))
    self.enterContext(patch.object(api_server, "session_registry", self.registry))
    self.enterContext(patch.object(api_server, "worker_manager", self.manager))
    self.enterContext(patch.object(api_server, "owner_locks", defaultdict(asyncio.Lock)))
    self.enterContext(patch.object(api_server, "OPEN_RL_VERSION", "v9.9.9"))
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "vllm", "OPEN_RL_ENABLE_FFT": "true", "VLLM_MAX_MODEL_LEN": "8192"}))

  async def asyncSetUp(self):
    self.client = await self.enterAsyncContext(asgi_client())

  async def get(self, path: str):
    return await self.client.get(f"/api/v1/{path}")

  async def test_capabilities_describe_the_deployment_without_breaking_the_sdk_fields(self):
    body = (await self.get("get_server_capabilities")).json()
    # The fields the SDK reads are untouched.
    self.assertIn("supported_models", body)
    self.assertIn("single_process", body)
    info = body["openrl"]
    self.assertEqual(info["version"], "v9.9.9")
    self.assertEqual(info["worker_launcher"], "process")
    self.assertTrue(info["fft_enabled"])
    self.assertTrue(info["models"]["any_huggingface_model"])
    self.assertEqual(info["limits"]["vllm_max_model_len"], 8192)
    self.assertEqual(info["limits"]["session_ttl_seconds"], 120)
    self.assertEqual(info["skill_url"], "https://github.com/gke-labs/open-rl/releases/download/v9.9.9/openrl-skill.tar.gz")
    self.assertEqual(info["endpoints"]["status"], "/api/v1/openrl/status")

  async def test_the_settings_catalog_follows_the_settings_model(self):
    catalog = {entry["name"]: entry for entry in api_server.settings_catalog()}
    self.assertEqual(set(catalog), {f"openrl.{name}" for name in api_server.Settings.model_fields})
    kind = catalog["openrl.fine_tuning_type"]
    self.assertEqual(kind["type"], "string")
    self.assertEqual(kind["allowed"], ["lora", "full"])
    self.assertIsNone(kind["default"])
    replicas = catalog["openrl.sampler_replicas"]
    self.assertEqual((replicas["type"], replicas["default"], replicas["min"], replicas["max"]), ("int", 1, 1, 8))
    self.assertEqual(catalog["openrl.exclusive"]["type"], "bool")
    self.assertEqual(catalog["openrl.weight_sync"]["allowed"], ["delta", "full"])
    for entry in catalog.values():
      self.assertTrue(entry["description"], f"{entry['name']} needs a description; the skill shows it")

  async def test_a_dev_build_has_no_skill_url(self):
    with patch.object(api_server, "OPEN_RL_VERSION", "sha-abc1234"):
      info = (await self.get("get_server_capabilities")).json()["openrl"]
    self.assertEqual(info["version"], "sha-abc1234")
    self.assertIsNone(info["skill_url"])

  async def test_status_lists_owners_their_sessions_and_workers(self):
    session = (await post_json(self.client, "create_session", {}))["session_id"]
    model_id = (await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    body = (await self.get("openrl/status")).json()
    self.assertEqual(body["version"], "v9.9.9")
    self.assertEqual(body["worker_launcher"], "process")
    self.assertTrue(body["store"]["ok"])
    self.assertEqual(body["sessions"]["live"], 1)
    self.assertIsNone(body["fleet"])
    (owner,) = body["owners"]
    self.assertEqual(owner["owner"], "test-base")
    self.assertEqual(owner["sessions"], [{"session_id": session, "live": True}])
    (worker,) = owner["workers"]
    self.assertEqual((worker["role"], worker["model_id"], worker["phase"], worker["ready"]), ("trainer", model_id, "Running", True))
    self.assertEqual(body["workers_by_phase"], {"Running": 1})
    self.assertIn("trainer fake-", worker["summary"])

  async def test_models_are_listed_newest_first_with_their_owner_and_session(self):
    session = (await post_json(self.client, "create_session", {}))["session_id"]
    first = (await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    second = (await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    body = (await self.get("openrl/models")).json()
    self.assertEqual([m["model_id"] for m in body["models"]], [second, first])
    entry = body["models"][0]
    self.assertEqual((entry["base_model"], entry["owner"], entry["fine_tuning_type"]), ("test-base", "test-base", "lora"))
    self.assertTrue(entry["session_live"])
    self.assertTrue(entry["shares_workers"])

  async def test_one_model_shows_settings_workers_and_a_summary(self):
    session = (await post_json(self.client, "create_session", {"user_metadata": {"openrl.fine_tuning_type": "full"}}))["session_id"]
    model_id = (await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    body = (await self.get(f"openrl/models/{model_id}")).json()
    self.assertEqual(body["settings"]["openrl.fine_tuning_type"], "full")
    self.assertEqual(body["settings"]["openrl.weight_sync"], "delta")
    self.assertFalse(body["shares_workers"])
    self.assertEqual(body["owner"], model_id)
    self.assertEqual(body["sessions"], [{"session_id": session, "live": True}])
    self.assertEqual(len(body["workers"]), 1)
    self.assertIn("trainer", body["summary"])

  async def test_a_released_model_says_so_in_its_summary(self):
    session = (await post_json(self.client, "create_session", {}))["session_id"]
    model_id = (await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session}))["request_id"]
    await self.state.delete_values(f"open_rl:session:{session}")
    self.manager.ensured.clear()  # the reaper removed the workers
    body = (await self.get(f"openrl/models/{model_id}")).json()
    self.assertFalse(body["session_live"])
    self.assertEqual(body["workers"], [])
    self.assertIn("released", body["summary"])

  async def test_unknown_model_and_session_are_404s_with_the_shared_error_shape(self):
    response = await self.get("openrl/models/nope")
    self.assertEqual(response.status_code, 404)
    self.assertIn("Unknown model", response.json()["error"])
    response = await self.get("openrl/sessions/sess-nope")
    self.assertEqual(response.status_code, 404)
    self.assertIn("expired", response.json()["error"])

  async def test_a_session_reports_live_settings_and_owners_then_expiry(self):
    session = (await post_json(self.client, "create_session", {"user_metadata": {"openrl.exclusive": "true"}}))["session_id"]
    await post_json(self.client, "create_model", {"base_model": "test-base", "session_id": session})
    body = (await self.get(f"openrl/sessions/{session}")).json()
    self.assertTrue(body["live"])
    self.assertEqual(body["user_metadata"], {"openrl.exclusive": "true"})
    self.assertEqual(len(body["owners"]), 1)
    self.assertEqual(body["ttl_seconds"], 120)
    # Expired but still attached: reported as not live, not as unknown.
    await self.state.delete_values(f"open_rl:session:{session}")
    body = (await self.get(f"openrl/sessions/{session}")).json()
    self.assertFalse(body["live"])
    self.assertEqual(body["user_metadata"], {})

  async def test_status_without_a_worker_manager_is_still_answered(self):
    with patch.object(api_server, "worker_manager", None):
      body = (await self.get("openrl/status")).json()
    self.assertEqual(body["worker_launcher"], "none")
    self.assertEqual(body["owners"], [])
    self.assertIsNone(body["fleet"])


def pod(name: str, phase: str, waiting: str | None = None, restarts: int = 0):
  state = SimpleNamespace(waiting=SimpleNamespace(reason=waiting) if waiting else None)
  statuses = [SimpleNamespace(state=state, restart_count=restarts, ready=phase == "Running")]
  return SimpleNamespace(metadata=SimpleNamespace(name=name), status=SimpleNamespace(phase=phase, container_statuses=statuses))


def node(name: str, labels: dict[str, str]):
  return SimpleNamespace(metadata=SimpleNamespace(name=name, labels=labels))


def gpu_slice(node_name: str, product: str, memory: str, count: int, driver: str = "gpu.nvidia.com"):
  devices = [
    SimpleNamespace(attributes={"productName": SimpleNamespace(string=product)}, capacity={"memory": SimpleNamespace(value=memory)})
    for _ in range(count)
  ]
  return SimpleNamespace(spec=SimpleNamespace(driver=driver, node_name=node_name, devices=devices))


class FakeCoreApi:
  def __init__(self, pods, nodes):
    self.pods, self.nodes = pods, nodes

  def list_namespaced_pod(self, namespace):
    return SimpleNamespace(items=self.pods)

  def list_node(self, label_selector):
    assert label_selector == "openrl.io/enabled=true"
    return SimpleNamespace(items=self.nodes)


class FakeResourceApi:
  def __init__(self, slices):
    self.slices = slices

  def list_resource_slice(self):
    return SimpleNamespace(items=self.slices)


def workload(name: str, role: str, owner: str, model_id: str, status: dict, created="2026-10-12T00:00:00Z", deleting=False):
  metadata = {"name": name, "creationTimestamp": created, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}}
  if deleting:
    metadata["deletionTimestamp"] = created
  return {"metadata": metadata, "spec": {"role": role, "ownerID": owner, "modelID": model_id}, "status": status}


class SchedulerViewsTest(unittest.TestCase):
  """The cluster side of the views: Workloads joined with pods, and the fleet."""

  def setUp(self):
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://test"}))
    self.custom = FakeCustomObjectsApi()
    self.custom.existing = {
      "lora-base-0-trainer": workload(
        "lora-base-0-trainer",
        "trainer",
        "base",
        "base",
        {"phase": "Running", "podName": "orw-t", "nodeName": "node-a", "deviceCount": 1, "reason": "PodCreated"},
      ),
      "lora-base-0-sampler": workload(
        "lora-base-0-sampler", "sampler", "base", "base", {"phase": "Placing", "podName": "orw-s", "nodeName": "node-b"}
      ),
      "fft-job-trainer": workload("fft-job-trainer", "trainer", "job", "job", {"phase": "Pending", "reason": "NoCapacity: needs 40Gi"}),
    }
    pods = [pod("orw-t", "Running"), pod("orw-s", "Pending", waiting="ContainerCreating")]
    nodes = [node("node-a", {"openrl.io/trainer": "true"}), node("node-b", {"openrl.io/sampler": "true"}), node("node-c", {})]
    slices = [
      gpu_slice("node-a", "NVIDIA L4", "22Gi", 2),
      gpu_slice("node-b", "NVIDIA L4", "22Gi", 2),
      gpu_slice("node-c", "NVIDIA H100 80GB HBM3", "81559Mi", 1),
    ]
    slices.append(gpu_slice("node-a", "domain", "0", 1, driver="compute-domain.nvidia.com"))
    self.manager = SchedulerWorkerManager(custom_api=self.custom, core_api=FakeCoreApi(pods, nodes), resource_api=FakeResourceApi(slices))

  def test_workers_join_workload_status_with_pod_state(self):
    by_name = {w["workload"]: w for w in self.manager.inspect_all()}
    trainer = by_name["lora-base-0-trainer"]
    self.assertEqual(
      (trainer["phase"], trainer["node"], trainer["pod_phase"], trainer["pod_waiting"], trainer["devices"]), ("Running", "node-a", "Running", None, 1)
    )
    self.assertGreater(trainer["age_seconds"], 0)
    sampler = by_name["lora-base-0-sampler"]
    self.assertEqual((sampler["phase"], sampler["pod_phase"], sampler["pod_waiting"]), ("Placing", "Pending", "ContainerCreating"))
    pending = by_name["fft-job-trainer"]
    self.assertEqual((pending["phase"], pending["pod"], pending["node"]), ("Pending", None, None))
    self.assertIn("NoCapacity", pending["reason"])
    self.assertEqual([w["workload"] for w in self.manager.inspect_owner("job")], ["fft-job-trainer"])

  def test_fleet_counts_devices_per_accelerator_and_our_running_workers(self):
    fleet = self.manager.fleet()
    nodes = {n["name"]: n for n in fleet["nodes"]}
    self.assertEqual(nodes["node-a"]["roles"], ["trainer"])
    self.assertEqual(nodes["node-c"]["roles"], ["trainer", "sampler"])  # no role label accepts both
    self.assertEqual((nodes["node-a"]["devices"], nodes["node-a"]["openrl_workers"]), (2, 1))
    self.assertEqual((nodes["node-b"]["devices"], nodes["node-b"]["openrl_workers"]), (2, 0))  # Placing is not Running
    by_product = {a["accelerator"]: a for a in fleet["accelerators"]}
    self.assertEqual((by_product["NVIDIA L4"]["devices"], by_product["NVIDIA L4"]["nodes"], by_product["NVIDIA L4"]["openrl_workers"]), (4, 2, 1))
    self.assertEqual(by_product["NVIDIA H100 80GB HBM3"]["memory"], "81559Mi")

  def test_fleet_read_errors_are_reported_not_raised(self):
    class Broken:
      def list_node(self, label_selector):
        raise RuntimeError("forbidden")

    manager = SchedulerWorkerManager(custom_api=self.custom, core_api=Broken(), resource_api=FakeResourceApi([]))
    self.assertIn("forbidden", manager.fleet()["error"])

  def test_views_are_cached_briefly(self):
    first = self.manager.inspect_all()
    self.custom.existing.clear()
    self.assertEqual(self.manager.inspect_all(), first)
    self.manager.view_cache_seconds = 0
    self.assertEqual(self.manager.inspect_all(), [])


if __name__ == "__main__":
  unittest.main()
